"""OpenAI-compatible proxy in front of Ollama, with API key management.

Security model, in one paragraph: every request that reaches Ollama must carry a
bearer key that is on the allow-list. The endpoints that *manage* that list
(`/api/keys`) are not protected by an ordinary client key — otherwise any client
could mint itself a new key and the allow-list would mean nothing. They require
the admin key instead, which is `ADMIN_API_KEY` from the environment or, when
that is unset, a random key generated on first start and persisted beside the
allow-list with mode 0600.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Optional, Dict, Any
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path

import requests

try:  # .env loading is a convenience; environment variables work without it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when python-dotenv is absent
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

log = logging.getLogger("ollama-proxy")

app = FastAPI(title="OpenAI to Ollama API Proxy", version="1.1.0")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "11433"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

# Where the allow-list and the generated admin key live. Kept out of the source
# tree by .gitignore; created with 0600 because it holds credentials.
KEYS_FILE = Path(os.getenv("KEYS_FILE", "api_keys.json"))


# ── key store ────────────────────────────────────────────────────────────────
def _load_store() -> Dict[str, Any]:
    if not KEYS_FILE.exists():
        return {"keys": [], "admin": None}
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("cannot read %s (%s); starting from an empty allow-list", KEYS_FILE, exc)
        return {"keys": [], "admin": None}
    return {"keys": list(data.get("keys") or []), "admin": data.get("admin")}


def _save_store() -> None:
    payload = {"keys": sorted(VALID_API_KEYS), "admin": ADMIN_API_KEY}
    tmp = KEYS_FILE.with_suffix(KEYS_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(KEYS_FILE)


def _new_key() -> str:
    # Secrets, not MD5-of-timestamp: an API key is a credential and must not be
    # guessable from the clock.
    return "sk-" + secrets.token_urlsafe(32)


_store = _load_store()
VALID_API_KEYS = {k for k in _store["keys"] if isinstance(k, str) and k}

# Environment wins; otherwise reuse whatever was generated last time so a
# restart does not lock the operator out of their own key management.
ADMIN_API_KEY = (os.getenv("ADMIN_API_KEY") or _store.get("admin") or "").strip()
if not ADMIN_API_KEY:
    ADMIN_API_KEY = _new_key()
    _save_store()


# ── auth ─────────────────────────────────────────────────────────────────────
def _bearer(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format, expected 'Bearer <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def get_api_key(auth_header: Optional[str] = Depends(API_KEY_HEADER)) -> str:
    """A client key: must be on the allow-list."""
    token = _bearer(auth_header)
    if not any(secrets.compare_digest(token, k) for k in VALID_API_KEYS):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def require_admin(auth_header: Optional[str] = Depends(API_KEY_HEADER)) -> str:
    """The admin key: the only thing allowed to change the allow-list."""
    token = _bearer(auth_header)
    if not secrets.compare_digest(token, ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ── schemas ──────────────────────────────────────────────────────────────────
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


class CreateKeyRequest(BaseModel):
    name: Optional[str] = None


def generate_id() -> str:
    return f"chatcmpl-{hashlib.sha256(secrets.token_bytes(16)).hexdigest()[:20]}"


def _ollama_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": bool(request.stream),
    }
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens:
        # OpenAI's max_tokens caps the *output*; Ollama's equivalent is num_predict.
        # (num_ctx, which this used to set, is the context window — a different thing.)
        options["num_predict"] = request.max_tokens
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        options["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        options["presence_penalty"] = request.presence_penalty
    if options:
        payload["options"] = options
    return payload


# ── OpenAI-compatible surface ────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models(api_key: str = Depends(get_api_key)):
    try:
        response = await run_in_threadpool(
            requests.get, f"{OLLAMA_BASE_URL}/api/tags", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")

    models = [
        {
            "id": m.get("name", ""),
            "object": "model",
            "created": 0,
            "owned_by": "ollama",
        }
        for m in response.json().get("models", [])
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, api_key: str = Depends(get_api_key)
):
    payload = _ollama_payload(request)
    chat_id = generate_id()
    created_at = int(time.time())
    url = f"{OLLAMA_BASE_URL}/api/chat"

    if request.stream:
        try:
            response = await run_in_threadpool(
                requests.post, url, json=payload, stream=True, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")

        def generate():
            # Starlette runs a sync iterator in a worker thread, so this does not
            # block the event loop.
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    done = bool(data.get("done"))
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": data.get("message", {}).get("content", ""),
                                },
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if done:
                        break
            finally:
                response.close()
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        response = await run_in_threadpool(
            requests.post, url, json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")

    data = response.json()
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)

    return ChatCompletionResponse(
        id=chat_id,
        created=created_at,
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=data.get("message", {}).get("content", ""),
                ),
                finish_reason="stop" if data.get("done", True) else None,
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


# ── key management (admin only) ──────────────────────────────────────────────
@app.post("/api/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: Optional[CreateKeyRequest] = None, admin: str = Depends(require_admin)
):
    new_key = _new_key()
    VALID_API_KEYS.add(new_key)
    _save_store()
    return {"api_key": new_key, "name": body.name if body else None}


@app.get("/api/keys")
async def list_api_keys(admin: str = Depends(require_admin)):
    # The keys themselves are credentials; return a masked form plus a prefix so
    # an operator can tell them apart without handing out the secrets again.
    return {
        "keys": [
            {"prefix": k[:11], "masked": k[:11] + "…" + k[-4:]} for k in sorted(VALID_API_KEYS)
        ]
    }


@app.delete("/api/keys/{api_key}")
async def delete_api_key(api_key: str, admin: str = Depends(require_admin)):
    for existing in list(VALID_API_KEYS):
        if secrets.compare_digest(existing, api_key):
            VALID_API_KEYS.discard(existing)
            _save_store()
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="API key not found")


@app.get("/health")
async def health_check():
    """Unauthenticated on purpose: this is what a load balancer probes."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    log.basicConfig(level=logging.INFO)
    if not os.getenv("ADMIN_API_KEY"):
        log.warning(
            "ADMIN_API_KEY is not set; using the generated admin key stored in %s",
            KEYS_FILE,
        )
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
