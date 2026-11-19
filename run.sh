#!/usr/bin/env bash
# run.sh — start the Telecom Churn Engine (API + website) with one command.
#
#   ./run.sh            # runs on port 8000
#   PORT=9000 ./run.sh  # pick another port
#
# Then open http://localhost:8000/ui/ for the website,
#        or http://localhost:8000/docs for the API docs.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

# Use the project virtualenv if there is one.
if [ -f myenv/bin/activate ]; then
  # shellcheck disable=SC1091
  source myenv/bin/activate
fi

# Install dependencies if FastAPI/uvicorn are missing (first run).
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

if [ ! -f models/lgbm_churn.pkl ]; then
  echo "⚠️  No trained model in models/. Train first:"
  echo "    python -m src.train --client data/Client.csv --record data/Record.csv --output models/"
fi

echo ""
echo "  Website:  http://localhost:${PORT}/ui/"
echo "  API docs: http://localhost:${PORT}/docs"
echo "  (Ctrl+C to stop)"
echo ""

exec uvicorn api.main:app --reload --port "$PORT"
