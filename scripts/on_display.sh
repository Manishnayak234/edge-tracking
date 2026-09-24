#!/usr/bin/env bash
# Runs a command with its windows on the Jetson's own display; works from an SSH session.
# Usage: scripts/on_display.sh <command> [args...]
#   e.g. scripts/on_display.sh ./build/detect_view
set -euo pipefail

# Use the logged-in desktop's X display (the login screen can't be used over SSH).
if [[ -z "${DISPLAY:-}" ]]; then
    DISPLAY=$(ls /tmp/.X11-unix/ 2>/dev/null | sed -n 's/^X/:/p' | tail -1)
    export DISPLAY
fi
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
if ! xset q >/dev/null 2>&1; then
    echo "Cannot open display ${DISPLAY:-none}: log in on the Jetson's desktop first." >&2
    exit 1
fi
exec "$@"
