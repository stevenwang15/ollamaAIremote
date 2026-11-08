# Ollama OpenAI Proxy with API Key Management

## Introduction

Ollama natively supports the OpenAI API schema, but it **lacks access authentication and API key management**. Exposing raw Ollama to the public network leads to unauthorized access and resource abuse.

This project is a lightweight FastAPI proxy layer for Ollama. It adds a security gateway with full OpenAI API compatibility: **API key authentication and key management**, while remaining compatible with OpenAI clients, SDKs and WebUI tools. It can be deployed on an intranet or a public server.

## Security model

Two kinds of credential, deliberately kept separate:

| Credential | What it can do |
| :--- | :--- |
| **Admin key** | Create, list and delete client keys. Cannot reach Ollama. |
| **Client key** | Call `/v1/models` and `/v1/chat/completions`. Cannot manage keys. |

That split is the whole point: if a client key could create keys, the allow-list would mean nothing, since any client could mint itself a new one and walk straight past the gate.

The admin key comes from `ADMIN_API_KEY`, or — when that is unset — is generated on first start, stored in `KEYS_FILE` (default `api_keys.json`, mode `0600`, gitignored) and reused on every restart. The allow-list is persisted in the same file, so client keys survive a restart.

## Features

- 🔑 **API key authentication & management** — admin-scoped create / list / delete; the allow-list is persisted, not in-memory only.
- 🔄 **OpenAI-compatible endpoints** — `/v1/models`, `/v1/chat/completions`.
- ⚡ **Streaming** — SSE chunks for `stream=true`, iterated off the event loop.
- 📊 **Real token usage** — `usage` is filled from Ollama's own `prompt_eval_count` / `eval_count` instead of being hardcoded to 0.
- ⚙️ **Env configuration** — Ollama address, listen host/port, timeout and keys via `.env`.
- 📦 **One-click start** — macOS / Linux / Windows; `start.sh` creates the venv and installs dependencies.
- 🧩 **Non-intrusive proxy** — pass-through to Ollama, no model logic changed.

## Prerequisites

- Python 3.9+
- A running Ollama service (default `http://localhost:11434`)

## Installation & start

```bash
git clone https://github.com/stevenwang15/ollamaAIremote.git
cd ollamaAIremote

cp .env.example .env
chmod +x start.sh
./start.sh
```

The proxy listens on `http://localhost:11433`.

## Configuration

Copy `.env.example` to `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
LISTEN_HOST=0.0.0.0
LISTEN_PORT=11433

# Admin key. Empty means "generate one on first start and persist it".
ADMIN_API_KEY=
KEYS_FILE=api_keys.json
REQUEST_TIMEOUT=300
```

> **Upgrading from 1.0:** the old `API_KEYS` whitelist is no longer read. Client keys now live in `KEYS_FILE` and are created through the admin-authenticated `/api/keys` endpoint.

## API endpoints

| Endpoint | Method | Auth | Description |
| :--- | :--- | :--- | :--- |
| `/health` | GET | none | Health probe, unauthenticated by design |
| `/v1/models` | GET | client key | List available Ollama models |
| `/v1/chat/completions` | POST | client key | Chat completion (normal or streaming) |
| `/api/keys` | POST | **admin key** | Create a client key |
| `/api/keys` | GET | **admin key** | List client keys (masked) |
| `/api/keys/{api_key}` | DELETE | **admin key** | Delete a client key |

## Usage

### Create a client key

```bash
curl -X POST http://localhost:11433/api/keys \
  -H "Authorization: Bearer $ADMIN_API_KEY"
# {"api_key":"sk-…","name":null}
```

### Chat request

```bash
curl -X POST http://localhost:11433/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CLIENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello"}]}'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11433/v1", api_key="YOUR_CLIENT_KEY")

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
print(response.usage)  # prompt / completion / total, from Ollama's own counters
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite covers the security property the project exists for — a client key must not be able to mint more client keys — plus persistence across restart, `max_tokens → num_predict`, and real token usage. It needs `requirements-dev.txt` (`httpx` for `TestClient`).

`./test.sh` is the complementary live smoke test against a running proxy; it requires `ADMIN_API_KEY`.

## Project structure

```
ollamaAIremote/
├── main.py
├── tests/test_main.py
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── start.sh
├── test.sh
├── LICENSE
└── README.md
```

## License

MIT
