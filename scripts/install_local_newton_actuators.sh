#!/usr/bin/env bash
# Install the sibling newton-actuators checkout as an editable package
# into this repo's uv-managed venv, replacing any PyPI-published copy.
#
# Usage:
#   scripts/install_local_newton_actuators.sh            # default sibling path
#   scripts/install_local_newton_actuators.sh /abs/path  # override path
#
# Notes:
#   - Only affects the local venv; pyproject.toml is untouched.
#   - Re-running `uv sync` may overwrite this editable install. Re-run
#     this script afterwards, or switch to `[tool.uv.sources]` for a
#     persistent binding.
set -euo pipefail

# Repo root of the *newton* checkout this script lives in.
NEWTON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Target newton-actuators checkout: CLI arg wins, else sibling dir.
ACTUATORS_DIR="${1:-$(cd "${NEWTON_DIR}/../newton-actuators" 2>/dev/null && pwd || true)}"

if [[ -z "${ACTUATORS_DIR}" || ! -d "${ACTUATORS_DIR}" ]]; then
    echo "error: newton-actuators source directory not found." >&2
    echo "       expected sibling at ${NEWTON_DIR}/../newton-actuators" >&2
    echo "       or pass an absolute path as the first argument." >&2
    exit 1
fi

if [[ ! -f "${ACTUATORS_DIR}/pyproject.toml" ]]; then
    echo "error: ${ACTUATORS_DIR} does not look like a Python project (no pyproject.toml)." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install it from https://docs.astral.sh/uv/." >&2
    exit 1
fi

cd "${NEWTON_DIR}"

echo "[install] newton repo:       ${NEWTON_DIR}"
echo "[install] actuators source:  ${ACTUATORS_DIR}"
echo

uv pip install -e "${ACTUATORS_DIR}"

echo
echo "[verify] importing newton_actuators..."
uv run python - <<'PY'
import newton_actuators
from newton_actuators import ActuatorPD

print(f"  module file : {newton_actuators.__file__}")
print(f"  version     : {newton_actuators.__version__}")
print(f"  ActuatorPD  : {ActuatorPD}")
PY

echo
echo "[done] newton_actuators now resolves to the local checkout."
