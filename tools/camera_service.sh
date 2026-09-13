#!/usr/bin/env bash
# Run the multi-camera AprilTag server as a launchd job on macOS.
#
# The plist is static: it runs `camera_service.sh foreground`, which discovers
# the attached cameras and pins each slot by uniqueID at every launch. A
# uniqueID embeds the USB location, so it changes whenever a camera moves
# ports; a list recorded in the plist would go stale the moment the rig is
# re-cabled. The server itself exits (code 75) when the rig changes under it
# -- a pinned camera leaves, a new one arrives, or an attached camera stops
# delivering -- and launchd relaunches it here, so re-cabling needs no hands.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${CAMERA_SERVICE_LABEL:-com.lukas.hexapod-cameras}"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="${CAMERA_SERVICE_LOG:-$HOME/Library/Logs/hexapod-cameras.log}"
PORT="${CAMERA_SERVICE_PORT:-8766}"
HOST="${CAMERA_SERVICE_HOST:-127.0.0.1}"
ROBOT_URL="${CAMERA_SERVICE_ROBOT_URL:-}"
EXTRA_ARGS="${CAMERA_SERVICE_EXTRA_ARGS:-}"
# Comma-separated uniqueIDs to skip. Two cameras on one USB controller cannot
# both stream; discovery excludes the loser itself and says so in the log, and
# this overrides its choice.
EXCLUDE="${CAMERA_SERVICE_EXCLUDE:-}"
# Per-camera intrinsics keyed by camera identity (stable_id / device_name), so
# the file survives slot renumbering. Empty runs planar-only.
CALIBRATION="${CAMERA_SERVICE_CALIBRATION-$ROOT/configs/camera_intrinsics_lab_20260912.json}"
UV_BIN="${UV:-$(command -v uv || echo /opt/homebrew/bin/uv)}"

usage() {
  cat <<EOF
Usage: tools/camera_service.sh <command>

  start       Write the launchd job from the current environment and start it
  stop        Stop and unload the job
  restart     Relaunch the job (rediscovers cameras; keeps the installed settings)
  status      launchctl state, port, and per-camera health
  logs        Tail $LOG
  cameras     Print the cameras a launch would pin, and exit
  foreground  Discover cameras and run the server in this shell

Env (read by start and baked into the plist):
  CAMERA_SERVICE_HOST=$HOST  CAMERA_SERVICE_PORT=$PORT
  CAMERA_SERVICE_ROBOT_URL, CAMERA_SERVICE_EXTRA_ARGS, CAMERA_SERVICE_LABEL,
  CAMERA_SERVICE_EXCLUDE=<uniqueID,uniqueID>
  CAMERA_SERVICE_CALIBRATION=<identity-keyed intrinsics json; "" for none>
    default: configs/camera_intrinsics_lab_20260912.json
EOF
}

rig() {
  local args=(--exclude "$EXCLUDE")
  [ -n "$CALIBRATION" ] && args+=(--calibration "$CALIBRATION")
  (cd "$ROOT" && OPENCV_AVFOUNDATION_SKIP_AUTH=1 "$UV_BIN" run python -m hexapod_tracker.rig "${args[@]}")
}

build_args() {
  local slots=() pins=() slot stable_id name
  while IFS=$'\t' read -r slot stable_id name; do
    [ -z "${slot:-}" ] && continue
    slots+=("$slot")
    pins+=("--device-id" "${slot}:${stable_id}")
  done < <(rig)
  [ "${#slots[@]}" -gt 0 ] || { echo "no rig cameras found" >&2; return 1; }
  printf '%s\n' "--indices" "${slots[@]}" "--native-avfoundation" "${slots[@]}" \
    "${pins[@]}" "--host" "$HOST" "--port" "$PORT" "--rig-exclude" "$EXCLUDE"
  [ -n "$ROBOT_URL" ] && printf '%s\n' "--robot-url" "$ROBOT_URL"
  if [ -n "$CALIBRATION" ]; then
    if [ -f "$CALIBRATION" ]; then
      printf '%s\n' "--camera-calibration" "$CALIBRATION"
    else
      echo "calibration file not found, running planar-only: $CALIBRATION" >&2
    fi
  fi
  # shellcheck disable=SC2086
  [ -n "$EXTRA_ARGS" ] && printf '%s\n' $EXTRA_ARGS
  return 0
}

write_plist() {
  local script="$ROOT/tools/camera_service.sh"
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict>'
    echo "  <key>Label</key><string>${LABEL}</string>"
    echo '  <key>ProgramArguments</key><array>'
    printf '    <string>%s</string>\n' /bin/bash "$script" foreground
    echo '  </array>'
    echo "  <key>WorkingDirectory</key><string>${ROOT}</string>"
    echo '  <key>EnvironmentVariables</key><dict>'
    echo '    <key>OPENCV_AVFOUNDATION_SKIP_AUTH</key><string>1</string>'
    for var in CAMERA_SERVICE_HOST CAMERA_SERVICE_PORT CAMERA_SERVICE_ROBOT_URL \
               CAMERA_SERVICE_EXTRA_ARGS CAMERA_SERVICE_EXCLUDE CAMERA_SERVICE_LABEL \
               CAMERA_SERVICE_LOG CAMERA_SERVICE_CALIBRATION; do
      case "$var" in
        CAMERA_SERVICE_HOST) value="$HOST" ;;
        CAMERA_SERVICE_PORT) value="$PORT" ;;
        CAMERA_SERVICE_ROBOT_URL) value="$ROBOT_URL" ;;
        CAMERA_SERVICE_EXTRA_ARGS) value="$EXTRA_ARGS" ;;
        CAMERA_SERVICE_EXCLUDE) value="$EXCLUDE" ;;
        CAMERA_SERVICE_LABEL) value="$LABEL" ;;
        CAMERA_SERVICE_LOG) value="$LOG" ;;
        CAMERA_SERVICE_CALIBRATION) value="$CALIBRATION" ;;
      esac
      printf '    <key>%s</key><string>%s</string>\n' "$var" "$(printf '%s' "$value" | sed 's/&/\&amp;/g; s/</\&lt;/g')"
    done
    echo '  </dict>'
    echo '  <key>RunAtLoad</key><true/>'
    echo '  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>'
    echo '  <key>ThrottleInterval</key><integer>10</integer>'
    echo "  <key>StandardOutPath</key><string>${LOG}</string>"
    echo "  <key>StandardErrorPath</key><string>${LOG}</string>"
    echo '</dict></plist>'
  } > "$PLIST"
}

loaded() { launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1; }

bootout_and_wait() {
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  local i
  for i in $(seq 1 50); do
    loaded || return 0
    sleep 0.2
  done
  echo "warning: ${LABEL} still loaded after bootout" >&2
}

bootstrap_with_retry() {
  # launchd answers "Bootstrap failed: 5: Input/output error" when asked to
  # load a label it is still tearing down; the job is fine a moment later.
  local i
  for i in $(seq 1 10); do
    if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
}

case "${1:-}" in
  cameras) rig ;;
  start)
    write_plist
    bootout_and_wait
    bootstrap_with_retry
    echo "started ${LABEL} on http://${HOST}:${PORT}/ (plist: ${PLIST})"
    ;;
  stop)
    bootout_and_wait
    echo "stopped ${LABEL}"
    ;;
  restart)
    if loaded; then
      launchctl kickstart -k "gui/$(id -u)/${LABEL}"
      echo "relaunched ${LABEL}"
    else
      "$0" start
    fi
    ;;
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
