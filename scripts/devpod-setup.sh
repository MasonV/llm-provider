#!/usr/bin/env bash
# devpod-setup.sh — Run by devpod after container creation/recreation.
# Installs llm-provider in editable mode so all optional deps are available.
set -euo pipefail

pip install -e /llm-provider[all]
