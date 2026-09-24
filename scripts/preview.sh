#!/usr/bin/env bash
# Live camera preview on the Jetson's own display; works from an SSH session.
# Usage: scripts/preview.sh [seconds]   (default: run until Ctrl+C)
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

FPS=60
BUFFERS=-1
if [[ $# -ge 1 ]]; then BUFFERS=$(( $1 * FPS )); fi

gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 sensor-mode=4 num-buffers="$BUFFERS" \
    ! "video/x-raw(memory:NVMM),width=1280,height=720,framerate=${FPS}/1,format=NV12" \
    ! nv3dsink sync=false 2>&1 | grep -E "ERROR|WARNING|Execution ended" || true
