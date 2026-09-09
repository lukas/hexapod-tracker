#!/usr/bin/env bash
# Run the multi-camera AprilTag server as a launchd job on macOS.
#
# Cameras are discovered and pinned by uniqueID at every start rather than
# recorded in the plist. A uniqueID embeds the USB location, so it changes
# whenever a camera moves ports -- a stored list goes stale the moment the rig
# is re-cabled, which is exactly how Robot Lab ended up configured for four
# cameras that no longer existed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${CAMERA_SERVICE_LABEL:-com.lukas.hexapod-cameras}"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="${CAMERA_SERVICE_LOG:-$HOME/Library/Logs/hexapod-cameras.log}"
PORT="${CAMERA_SERVICE_PORT:-8766}"
HOST="${CAMERA_SERVICE_HOST:-127.0.0.1}"
ROBOT_URL="${CAMERA_SERVICE_ROBOT_URL:-}"
EXTRA_ARGS="${CAMERA_SERVICE_EXTRA_ARGS:-}"
# Comma-separated uniqueIDs to skip. Two cameras on one USB controller
# cannot both stream, and the loser retries forever, taking bandwidth from
# the cameras that do work -- so excluding it is better than letting it
# thrash until the cable moves.
EXCLUDE="${CAMERA_SERVICE_EXCLUDE:-}"
UV_BIN="${UV:-$(command -v uv || echo /opt/homebrew/bin/uv)}"

usage() {
  cat <<EOF
Usage: tools/camera_service.sh <command>

  start     Install the launchd job and start it (idempotent)
  stop      Stop and unload the job
  restart   Stop, then start
  status    launchctl state, port, and per-camera health
  logs      Tail $LOG
  cameras   Print the cameras that a start would pin, and exit
  foreground  Run in this shell with the same arguments

Env: CAMERA_SERVICE_PORT, CAMERA_SERVICE_HOST, CAMERA_SERVICE_ROBOT_URL,
     CAMERA_SERVICE_EXTRA_ARGS, CAMERA_SERVICE_LABEL,
     CAMERA_SERVICE_EXCLUDE=<uniqueID,uniqueID>
EOF
}

# Real cameras only: the Studio Display and a Continuity iPhone are not part of
# the rig and would waste a slot and a worker.
discover() {
  cd "$ROOT"
  CAMERA_SERVICE_EXCLUDE="$EXCLUDE" OPENCV_AVFOUNDATION_SKIP_AUTH=1 "$UV_BIN" run python - <<'PY'
import sys
sys.path.insert(0, "src")
try:
    from hexapod_tracker.avfoundation_capture import AVFoundationYuvCapture as C
    devices = C.device_descriptors()
except Exception as error:  # never emit a half-built argument list
    print(f"discovery failed: {error}", file=sys.stderr)
    raise SystemExit(1)
import os

excluded = {
    value.strip()
    for value in os.environ.get("CAMERA_SERVICE_EXCLUDE", "").split(",")
    if value.strip()
}
wanted = [
    item for item in devices
    if item["available"] and item["kind"] == "external"
    and "studio display" not in item["name"].lower()
    and item["stable_id"] not in excluded
]
if not wanted:
    print("no rig cameras found", file=sys.stderr)
    raise SystemExit(1)
for slot, item in enumerate(wanted):
    print(f"{slot}\t{item['index']}\t{item['stable_id']}\t{item['name']}")
PY
}

build_args() {
  local slots=() indices=() pins=()
  while IFS=$'\t' read -r slot index stable_id name; do
    [ -z "${slot:-}" ] && continue
    slots+=("$index")
    pins+=("--device-id" "${index}:${stable_id}")
  done < <(discover)
  printf '%s\n' "--indices" "${slots[@]}" "--native-avfoundation" "${slots[@]}" \
    "${pins[@]}" "--host" "$HOST" "--port" "$PORT"
  [ -n "$ROBOT_URL" ] && printf '%s\n' "--robot-url" "$ROBOT_URL"
  # shellcheck disable=SC2086
  [ -n "$EXTRA_ARGS" ] && printf '%s\n' $EXTRA_ARGS
  return 0
}

write_plist() {
  local args=() line
  while IFS= read -r line; do args+=("$line"); done < <(build_args)
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict>'
    echo "  <key>Label</key><string>${LABEL}</string>"
    echo '  <key>ProgramArguments</key><array>'
    printf '    <string>%s</string>\n' "$UV_BIN" run hexapod-camera-server
    printf '    <string>%s</string>\n' "${args[@]}"
    echo '  </array>'
    echo "  <key>WorkingDirectory</key><string>${ROOT}</string>"
    echo '  <key>EnvironmentVariables</key><dict>'
    echo '    <key>OPENCV_AVFOUNDATION_SKIP_AUTH</key><string>1</string>'
    echo '  </dict>'
    echo '  <key>RunAtLoad</key><true/>'
    echo '  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>'
    echo '  <key>ThrottleInterval</key><integer>10</integer>'
    echo "  <key>StandardOutPath</key><string>${LOG}</string>"
    echo "  <key>StandardErrorPath</key><string>${LOG}</string>"
    echo '</dict></plist>'
  } > "$PLIST"
}

case "${1:-}" in
  cameras) discover ;;
  start)
    write_plist
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "started ${LABEL} on http://${HOST}:${PORT}/"
    ;;
  stop)
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    echo "stopped ${LABEL}"
    ;;
  restart) "$0" stop; sleep 2; "$0" start ;;
  status)
    launchctl list | grep -E "PID|${LABEL}" || echo "job not loaded"
    curl -s -m 5 "http://127.0.0.1:${PORT}/api/cameras/health" > /tmp/.camera_health.json \
      && "$UV_BIN" run python "$ROOT/tools/print_camera_health.py" /tmp/.camera_health.json \
      || echo "port ${PORT} not answering"
    ;;
  logs) tail -f "$LOG" ;;
  foreground)
    cd "$ROOT"
    args=(); while IFS= read -r line; do args+=("$line"); done < <(build_args)
    OPENCV_AVFOUNDATION_SKIP_AUTH=1 exec "$UV_BIN" run hexapod-camera-server "${args[@]}"
    ;;
  *) usage; exit 2 ;;
esac
