#!/usr/bin/env bash
# Stops any locally running llama-server started by scripts/serve.sh.
pkill -f "runtime/llama.cpp/.*/llama-server" && echo "stopped" || echo "no server running"
