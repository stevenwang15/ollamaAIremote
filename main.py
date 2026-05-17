from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import json
import hashlib
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="OpenAI to Ollama API Proxy", version="1.0.0")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "11433"))
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

api_keys_env = os.getenv("API_KEYS", "")
VALID_API_KEYS = set(k.strip() for k in api_keys_env.split(",") if k.strip())

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

def generate_id() -> str:
    return f"chatcmpl-{hashlib.md5(f'{time.time()}{hash(time.time())}'.encode()).hexdigest()[:20]}"

def get_current_timestamp() -> int:
    return int(time.time())

async def get_api_key(auth_header: Optional[str] = Depends(API_KEY_HEADER)) -> str:
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )
    api_key = auth_header[7:]
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

@app.get("/v1/models")
async def list_models(api_key: str = Depends(get_api_key)):
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        ollama_models = response.json()

        models = []
        for model in ollama_models.get("models", []):
            models.append({
                "id": model.get("name", ""),
                "object": "model",
                "created": 0,
                "owned_by": "ollama"
            })

        return {"object": "list", "data": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
    try:
        ollama_messages = []
        for msg in request.messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        payload = {
            "model": request.model,
            "messages": ollama_messages,
            "temperature": request.temperature,
            "stream": request.stream
        }

        if request.max_tokens:
            payload["options"] = {"num_ctx": request.max_tokens}
        if request.top_p:
            payload["options"] = payload.get("options", {})
            payload["options"]["top_p"] = request.top_p

        chat_id = generate_id()
        created_time = get_current_timestamp()

        if request.stream:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                stream=True
            )
            response.raise_for_status()

            def generate():
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": data.get('message', {}).get('content', '')
                                    },
                                    "finish_reason": "stop" if data.get("done") else None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass
                yield "data: [DONE]\n\n"

            from fastapi.responses import StreamingResponse
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            return ChatCompletionResponse(
                id=chat_id,
                created=created_time,
                model=request.model,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=data.get('message', {}).get('content', '')
                    ),
                    finish_reason="stop"
                )],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/keys")
async def create_api_key():
    new_key = hashlib.md5(f"{time.time()}{hash(time.time())}".encode()).hexdigest()
    VALID_API_KEYS.add(new_key)
    return {"api_key": new_key}

@app.get("/api/keys")
async def list_api_keys():
    return {"keys": list(VALID_API_KEYS)}

@app.delete("/api/keys/{api_key}")
async def delete_api_key(api_key: str):
    if api_key in VALID_API_KEYS:
        VALID_API_KEYS.remove(api_key)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="API key not found")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)