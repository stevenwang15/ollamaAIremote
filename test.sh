#!/bin/bash

echo "=========================================="
echo "  Testing Ollama API Proxy"
echo "=========================================="
echo ""

echo "[1/4] Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:11433/health)
echo "Response: $HEALTH_RESPONSE"

echo ""
echo "[2/4] Creating test API key..."
API_KEY=$(curl -s -X POST http://localhost:11433/api/keys | grep -o '"api_key":"[^"]*"' | cut -d'"' -f4)
echo "API Key: $API_KEY"

echo ""
echo "[3/4] Listing models..."
curl -s http://localhost:11433/v1/models -H "Authorization: Bearer $API_KEY" | head -100

echo ""
echo "[4/4] Testing chat completion..."
curl -s -X POST http://localhost:11433/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 50
  }' | head -100

echo ""
echo "=========================================="
echo "  Test completed!"
echo "=========================================="