# Ollama OpenAI Proxy with API Key Management
## Introduction
Ollama natively supports the OpenAI API schema, but it **lacks access authentication and API key management**. Exposing raw Ollama to the public network will lead to unauthorized access and resource abuse.

This project is a lightweight FastAPI proxy layer for Ollama. It adds a security gateway with full OpenAI API compatibility. You get **API key authentication and key management**, while remaining fully compatible with all OpenAI clients, SDKs and WebUI tools. It can be safely deployed on intranet or public servers.

## Features
- 🔑 **API Key Authentication & Management**  
  Built-in key whitelist, support create / list / delete API keys, prevent anonymous access and abuse.
- 🔄 **Full OpenAI API Compatibility**  
  Drop-in replacement for official OpenAI endpoints: `/v1/chat/completions`, `/v1/models`.
- ⚡ **Streaming Response Support**  
  Fully compatible with `stream=true` streaming output for real-time chat UI.
- ⚙️ **Simple Env Configuration**  
  Configure Ollama address, listen host, port and authorized keys via `.env`.
- 📦 **Cross-Platform One-Click Startup**  
  Works on macOS / Linux / Windows; auto virtual environment and dependency install via startup script.
- 🧩 **Non-intrusive Proxy Forwarding**  
  Direct pass-through to original Ollama backend, no modification to Ollama model logic.

## Prerequisites
- Python 3.8+
- Running Ollama service (default: `http://localhost:11434`)

## Installation & Start
```bash
git clone https://github.com/stevenwang-cn/ollamaAIremote.git
cd ollamaAIremote

chmod +x start.sh
./start.sh
```
Proxy service runs at: `http://localhost:11433`

## Configuration
Copy `.env.example` to `.env` and customize:
```env
OLLAMA_BASE_URL=http://localhost:11434
LISTEN_HOST=0.0.0.0
LISTEN_PORT=11433
API_KEYS=key1,key2,key3
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/v1/models` | GET | List available Ollama models |
| `/v1/chat/completions` | POST | Chat completion (normal/stream) |
| `/api/keys` | POST | Generate new API key |
| `/api/keys` | GET | List all keys |
| `/api/keys/{api_key}` | DELETE | Remove specified key |

## Usage Example

### Curl Chat Request
```bash
curl -X POST http://localhost:11433/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

### OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11433/v1",
    api_key="YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

## Project Structure
```
ollamaAIremote/
├── main.py
├── .env / .env.example
├── requirements.txt
├── start.sh
├── test.sh
├── LICENSE
└── README.md
```

## License
MIT License
