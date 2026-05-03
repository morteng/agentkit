#!/usr/bin/env bash
set -euo pipefail
uv sync --all-extras --group dev
uv run pre-commit install
uv run pytest tests/unit -x
echo "✓ ready"
