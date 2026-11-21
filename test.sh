#!/bin/bash
# Live smoke test against a running proxy.
#
#   ADMIN_API_KEY=... ./test.sh
#   BASE_URL=http://host:11433 ADMIN_API_KEY=... ./test.sh
#
# Creating a client key requires the admin key — that is the point of the design,
# so this script refuses to run without one.

set -u
BASE_URL=${BASE_URL:-http://localhost:11433}

if [ -z "${ADMIN_API_KEY:-}" ]; then
  echo "ADMIN_API_KEY is not set." >&2
  echo "Use the value from your .env, or the one in api_keys.json if it was generated." >&2
  exit 1
fi

printf '%s\n' "==========================================" "  Testing Ollama API Proxy" "=========================================="

echo
echo "[1/5] health endpoint"
if ! HEALTH=$(curl -sf --max-time 10 "$BASE_URL/health"); then
  echo "FAIL — proxy not reachable at $BASE_URL (start it with ./start.sh)" >&2
  exit 1
fi
echo "$HEALTH"

echo
echo "[2/5] key management must reject anonymous callers"
code=$(curl -s --max-time 10 -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/keys")
if [ "$code" = "401" ]; then
  echo "ok — anonymous key creation refused (401)"
else
  echo "FAIL — anonymous key creation returned $code, expected 401" >&2
  exit 1
fi

echo
echo "[3/5] create a client key as admin"
RESP=$(curl -s --max-time 10 -X POST "$BASE_URL/api/keys" \
  -H "Authorization: Bearer $ADMIN_API_KEY")
API_KEY=$(printf '%s' "$RESP" | sed -n 's/.*"api_key":"\([^"]*\)".*/\1/p')
if [ -z "$API_KEY" ]; then
  echo "FAIL — could not create a key: $RESP" >&2
  exit 1
fi
echo "ok — key $(printf '%s' "$API_KEY" | cut -c1-11)… created"

echo
echo "[4/5] list models with that key"
curl -s --max-time 30 "$BASE_URL/v1/models" -H "Authorization: Bearer $API_KEY" | head -40

echo
echo "[5/5] chat completion"
OUT=$(curl -s --max-time 120 -X POST "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello!"}],"stream":false,"max_tokens":50}' \
  -w '\n%{http_code}')
CODE=$(printf '%s\n' "$OUT" | tail -n 1)
printf '%s\n' "$OUT" | sed '$d' | head -40
if [ "$CODE" != "200" ]; then
  echo "FAIL — chat completion returned $CODE" >&2
  exit 1
fi

echo
echo "=========================================="
echo "  Done. Delete the test key when finished:"
echo "  curl -X DELETE $BASE_URL/api/keys/<key> -H 'Authorization: Bearer \$ADMIN_API_KEY'"
echo "=========================================="
