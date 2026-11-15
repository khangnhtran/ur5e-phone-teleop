#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
if [[ -e .venv-receiver ]]; then
    echo ".venv-receiver already exists; leaving it unchanged."
    exit 0
fi
"${RECEIVER_SYSTEM_PYTHON:-python3.10}" -m venv .venv-receiver
.venv-receiver/bin/python -I -m pip install -r requirements-receiver.txt
