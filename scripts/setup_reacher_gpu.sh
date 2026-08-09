#!/usr/bin/env bash
set -euo pipefail

# Install the Reacher stack in GeneralContainer's Python 3.13 environment.
# labmaze has no cp313 wheel and is not used by Reacher, so dm-control is
# installed without that optional transitive build after its required runtime
# dependencies. Python <=3.12 can use requirements-reacher.txt directly.

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libegl1 libgl1-mesa-glx libosmesa6 libglfw3 >/dev/null
fi

if "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 13))'; then
  grep -v '^dm-control' requirements-reacher.txt > /tmp/whitehole-reacher.txt
  "$PYTHON_BIN" -m pip install -r /tmp/whitehole-reacher.txt
  "$PYTHON_BIN" -m pip install \
    absl-py dm-env dm-tree glfw lxml 'mujoco>=3.11.0' protobuf \
    pyopengl pyparsing requests setuptools
  "$PYTHON_BIN" -m pip install --no-deps dm-control
else
  "$PYTHON_BIN" -m pip install -r requirements-reacher.txt
fi
