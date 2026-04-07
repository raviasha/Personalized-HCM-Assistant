#!/usr/bin/env bash
# setup.sh — create venv, install deps, apply compatibility patches, run app
set -e

PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11

echo "==> Creating virtual environment with Python 3.11..."
$PYTHON -m venv .venv

echo "==> Installing dependencies..."
.venv/bin/pip install --quiet \
  'gradio==4.44.1' \
  'huggingface_hub==0.24.7' \
  'starlette<1.0' \
  'openai>=1.40.0' \
  'chromadb>=0.5.0' \
  'tiktoken>=0.7.0' \
  'python-dotenv>=1.0.0'

echo "==> Applying gradio_client bool-schema patch..."
UTILS=".venv/lib/python3.11/site-packages/gradio_client/utils.py"

# Patch get_type: handle boolean schemas
if ! grep -q "isinstance(schema, bool)" "$UTILS"; then
  sed -i '' 's/^def get_type(schema: dict):$/def get_type(schema: dict):\n    if isinstance(schema, bool):\n        return "any"/' "$UTILS"
fi

# Patch _json_schema_to_python_type: handle boolean schemas
if ! grep -q "isinstance(schema, bool)" "$UTILS"; then
  sed -i '' 's/^    if schema == {}:$/    if isinstance(schema, bool):\n        return "Any"\n    if schema == {}:/' "$UTILS"
fi || true  # second grep may not find it if first already added it

echo "==> Patches applied."
echo ""
echo "==> Setup complete. Run the app with:"
echo "    .venv/bin/python app.py"
