#!/usr/bin/env bash
# Sends the challenge's example scenarios to a running instance and prints the answers.
set -euo pipefail
URL="${URL:-http://localhost:8000}"
USER_ID="${USER_ID:-cliente1988}"
messages=(
  "What's the difference between the Get Clássica and the Get Smart?"
  "What's the weather forecast in Porto Alegre tomorrow?"
  "When will the money from yesterday's sales be deposited?"
  "Do I need a bank account to receive my sales via Pix?"
  "My card machine won't connect to the internet, what should I do?"
  "How does receivables advance (antecipação) work with Getnet?"
  "What's the euro exchange rate today?"
  "My card machine is showing a transaction decline error."
  "How many installments can I split a sale into with the crediário?"
  "Can I sell through WhatsApp using the Payment Link?"
)
for m in "${messages[@]}"; do
  echo "=================================================================="
  echo "> $m"
  python3 - "$URL" "$m" "$USER_ID" <<'PY'
import json, sys, urllib.request

url, message, user_id = sys.argv[1:4]
req = urllib.request.Request(
    f"{url}/chat",
    data=json.dumps({"message": message, "user_id": user_id}).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=120))
tools = [c["tool"] for c in r.get("tool_calls", [])]
plan = (r.get("route") or {}).get("plan")
print(f"[{r['agent']} | plan={plan} | tools={tools} | {r['latency_ms']} ms | mode={r['mode']}]")
print(r["response"])
PY
done
