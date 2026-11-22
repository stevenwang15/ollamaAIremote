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
from typing import List, Optional, Dict, Any, Union
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

app = FastAPI(title="OpenAI to Ollama API Proxy", version="1.2.0")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")


def _env_number(name: str, default: str, cast):
    # A typo in .env should name the variable, not dump a traceback. An empty
    # value means "unset", the same convention .env.example uses for keys.
    raw = os.getenv(name) or default
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a valid {cast.__name__}, got {raw!r}")


LISTEN_PORT = _env_number("LISTEN_PORT", "11433", int)
REQUEST_TIMEOUT = _env_number("REQUEST_TIMEOUT", "300", float)
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

# Where the allow-list and the generated admin key live. Kept out of the source
# tree by .gitignore; created with 0600 because it holds credentials.
KEYS_FILE = Path(os.getenv("KEYS_FILE", "api_keys.json"))
if not KEYS_FILE.is_absolute():
    # Anchor relative paths beside this file, not the caller's cwd: starting
    # the proxy from another directory must not silently scatter (and lose)
    # the key store.
    KEYS_FILE = Path(__file__).resolve().parent / KEYS_FILE


# ── key store ────────────────────────────────────────────────────────────────
def _load_store() -> Dict[str, Any]:
    if not KEYS_FILE.exists():
        return {"keys": [], "admin": None, "meta": {}}
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("cannot read %s (%s); starting from an empty allow-list", KEYS_FILE, exc)
        return {"keys": [], "admin": None, "meta": {}}
    # A hand-edited or mangled file must not crash startup, and must never
    # degrade into single-character "keys" that would authenticate anything.
    if not isinstance(data, dict):
        log.warning("%s is not a JSON object; starting from an empty allow-list", KEYS_FILE)
        return {"keys": [], "admin": None, "meta": {}}
    keys = data.get("keys")
    admin = data.get("admin")
    # `meta` is new in 1.2; a file written by an older version has no such key
    # and must still load.
    meta = data.get("meta")
    return {
        "keys": keys if isinstance(keys, list) else [],
        "admin": admin if isinstance(admin, str) else None,
        "meta": meta if isinstance(meta, dict) else {},
    }


def _save_store() -> None:
    payload = {"keys": sorted(VALID_API_KEYS), "admin": ADMIN_API_KEY, "meta": KEY_META}
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
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
# Per-key bookkeeping: the optional name an operator gave a key, and when it was
# issued. The `name` used to be echoed back once and then thrown away.
KEY_META: Dict[str, Dict[str, Any]] = {
    key: value for key, value in _store["meta"].items() if isinstance(value, dict)
}

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
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format, expected 'Bearer <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _key_matches(a: str, b: str) -> bool:
    # compare_digest() raises TypeError on non-ASCII str, which would turn a
    # junk header into a 500; compare UTF-8 bytes instead.
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


async def get_api_key(auth_header: Optional[str] = Depends(API_KEY_HEADER)) -> str:
    """A client key: must be on the allow-list."""
    token = _bearer(auth_header)
    if not any(_key_matches(token, k) for k in VALID_API_KEYS):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def require_admin(auth_header: Optional[str] = Depends(API_KEY_HEADER)) -> str:
    """The admin key: the only thing allowed to change the allow-list."""
    token = _bearer(auth_header)
    if not _key_matches(token, ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ── schemas ──────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    # Optional because an assistant message that only carries tool_calls has
    # no text of its own; OpenAI sends `content: null` in that case.
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # newer OpenAI SDKs send this
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None  # {"include_usage": true}
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None  # accepted for compatibility; Ollama has no equivalent


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"  # Ollama can report done=false


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


def _ollama_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Forward the fields Ollama understands, including tool plumbing.

    A plain chat message used to be reduced to {role, content}; that silently
    dropped `tool_calls` / `tool_call_id`, so a client that ran a tool and sent
    the result back got a conversation with no tool history at all.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        item: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            item["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        if m.name:
            item["name"] = m.name
        out.append(item)
    return out


def _openai_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Ollama returns `arguments` as an object; OpenAI clients expect a JSON
    *string* plus an id and `type: "function"`. Passing the object through
    unchanged breaks every SDK that json-parses the arguments."""
    calls: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return calls
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        arguments = fn.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        calls.append(
            {
                "id": tc.get("id") or f"call_{secrets.token_hex(8)}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": arguments},
            }
        )
    return calls


def _token_usage(data: Dict[str, Any]) -> Dict[str, int]:
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _ollama_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": request.model,
        "messages": _ollama_messages(request.messages),
        "stream": bool(request.stream),
    }
    if request.tools:
        # Ollama uses the same {type, function} shape as OpenAI, so this is a
        # pass-through rather than a translation.
        payload["tools"] = request.tools
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        # Ollama only reads sampling parameters from `options`; a top-level
        # `temperature` on /api/chat is ignored, so it used to be a no-op.
        options["temperature"] = request.temperature
    max_output = request.max_tokens or request.max_completion_tokens
    if max_output:
        # OpenAI's max_tokens caps the *output*; Ollama's equivalent is num_predict.
        # (num_ctx, which this used to set, is the context window — a different thing.)
        options["num_predict"] = max_output
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.stop is not None:
        options["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if request.seed is not None:
        options["seed"] = request.seed
    if request.frequency_penalty is not None:
        options["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        options["presence_penalty"] = request.presence_penalty
    if options:
        payload["options"] = options
    return payload


def _ollama_error(exc: requests.exceptions.RequestException) -> HTTPException:
    """Turn a backend failure into an honest client-facing error: Ollama's own
    message when it sent one, and its 4xx status when the request was the
    problem (a bad model name is not "unreachable")."""
    response = getattr(exc, "response", None)
    if response is None:
        return HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")
    try:
        message = response.json().get("error")
    except (ValueError, AttributeError):
        message = None
    detail = f"Ollama error: {message}" if message else f"Ollama error: {exc}"
    status_code = response.status_code if 400 <= response.status_code < 500 else 502
    return HTTPException(status_code=status_code, detail=detail)


# ── OpenAI-compatible surface ────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models(api_key: str = Depends(get_api_key)):
    try:
        response = await run_in_threadpool(
            requests.get, f"{OLLAMA_BASE_URL}/api/tags", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise _ollama_error(exc)
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unexpected response; check OLLAMA_BASE_URL",
        )

    models = [
        {
            "id": m.get("name", ""),
            "object": "model",
            "created": 0,
            "owned_by": "ollama",
        }
        for m in body.get("models", [])
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
        response = None
        try:
            response = await run_in_threadpool(
                requests.post, url, json=payload, stream=True, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            # Build the error first — it may need to read the response body —
            # then close so the unread stream does not leak the connection.
            error = _ollama_error(exc)
            if response is not None:
                response.close()
            raise error

        include_usage = bool(
            isinstance(request.stream_options, dict)
            and request.stream_options.get("include_usage")
        )

        def generate():
            # Starlette runs a sync iterator in a worker thread, so this does not
            # block the event loop.
            usage = None
            seen_tool_calls = False
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        # Ollama reports some failures (bad model, load error)
                        # as an error object inside the stream itself.
                        yield f"data: {json.dumps({'error': {'message': data['error']}})}\n\n"
                        break
                    done = bool(data.get("done"))
                    message = data.get("message") or {}
                    tool_calls = _openai_tool_calls(message.get("tool_calls"))
                    if tool_calls:
                        seen_tool_calls = True
                    delta: Dict[str, Any] = {
                        "role": "assistant",
                        "content": message.get("content", ""),
                    }
                    if tool_calls:
                        # Ollama emits the whole call list at once; OpenAI's
                        # streaming schema is an array of indexed deltas.
                        delta["tool_calls"] = [
                            {"index": i, **tc} for i, tc in enumerate(tool_calls)
                        ]
                    if done:
                        finish_reason = "tool_calls" if seen_tool_calls else "stop"
                    else:
                        finish_reason = None
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if done:
                        usage = _token_usage(data)
                        break
            except requests.exceptions.RequestException as exc:
                # Connection dropped mid-stream: log it and end without [DONE]
                # so the client can tell the stream was truncated.
                log.warning("stream from Ollama aborted: %s", exc)
                return
            finally:
                response.close()
            # After the finally, not in it: yielding while a GeneratorExit is
            # in flight (client disconnect) raises RuntimeError.
            #
            # `stream_options.include_usage` asks for one extra chunk carrying
            # usage and an empty choices array, before [DONE].
            if include_usage and usage is not None:
                usage_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": request.model,
                    "choices": [],
                    "usage": usage,
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        response = await run_in_threadpool(
            requests.post, url, json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise _ollama_error(exc)

    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unexpected response; check OLLAMA_BASE_URL",
        )
    message = data.get("message") or {}
    tool_calls = _openai_tool_calls(message.get("tool_calls"))
    content = message.get("content") or ""
    if tool_calls:
        # OpenAI sends `content: null` when the turn is only a tool call.
        content = content or None
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop" if data.get("done", True) else None

    response = ChatCompletionResponse(
        id=chat_id,
        created=created_at,
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_token_usage(data),
    )
    # OpenAI keeps `content: null` on a tool-only turn but omits the fields that
    # do not apply; emitting `"tool_call_id": null` on a plain reply is noise
    # that strict clients reject.
    body = response.model_dump()
    for choice in body["choices"]:
        choice["message"] = {
            key: value
            for key, value in choice["message"].items()
            if value is not None or key == "content"
        }
    return body


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
    """Embeddings are what RAG clients (LlamaIndex, LangChain, vector stores)
    hit first; without this endpoint the proxy could only serve chat."""
    inputs = [request.input] if isinstance(request.input, str) else list(request.input)
    if not inputs or any(not isinstance(text, str) for text in inputs):
        raise HTTPException(
            status_code=400, detail="input must be a string or a non-empty list of strings"
        )

    try:
        response = await run_in_threadpool(
            requests.post,
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": request.model, "input": inputs},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise _ollama_error(exc)

    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unexpected response; check OLLAMA_BASE_URL",
        )

    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned no embeddings; check that the model supports embeddings",
        )

    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "model": request.model,
        # OpenAI reports tokens for the whole batch in one number.
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# ── key management (admin only) ──────────────────────────────────────────────
@app.post("/api/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: Optional[CreateKeyRequest] = None, admin: str = Depends(require_admin)
):
    new_key = _new_key()
    VALID_API_KEYS.add(new_key)
    name = body.name if body else None
    KEY_META[new_key] = {"name": name, "created": int(time.time())}
    _save_store()
    return {"api_key": new_key, "name": name}


@app.get("/api/keys")
async def list_api_keys(admin: str = Depends(require_admin)):
    # The keys themselves are credentials; return a masked form plus a prefix so
    # an operator can tell them apart without handing out the secrets again.
    return {
        "keys": [
            {
                "prefix": k[:11],
                "masked": k[:11] + "…" + k[-4:],
                "name": (KEY_META.get(k) or {}).get("name"),
                "created": (KEY_META.get(k) or {}).get("created"),
            }
            for k in sorted(VALID_API_KEYS)
        ]
    }


@app.delete("/api/keys/{api_key}")
async def delete_api_key(api_key: str, admin: str = Depends(require_admin)):
    for existing in list(VALID_API_KEYS):
        if _key_matches(existing, api_key):
            VALID_API_KEYS.discard(existing)
            KEY_META.pop(existing, None)
            _save_store()
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="API key not found")


@app.get("/health")
async def health_check():
    """Unauthenticated on purpose: this is what a load balancer probes."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # `log` is a Logger instance; basicConfig lives on the module.
    logging.basicConfig(level=logging.INFO)
    if not os.getenv("ADMIN_API_KEY"):
        log.warning(
            "ADMIN_API_KEY is not set; using the generated admin key stored in %s",
            KEYS_FILE,
        )
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
