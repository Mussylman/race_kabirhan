#!/usr/bin/env bash
# Race Vision — launch / stop / monitor the full stack.
#
# Services (in start order):
#   1. DeepStream pipeline (deepstream.main)  →  /dev/shm/rv_detections
#   2. Backend API + ResolutionEnforcer        →  ws://localhost:8000/ws
#   3. Frontend (Vite dev server)              →  http://localhost:5173
#
# Usage:
#   scripts/rv.sh start        # boot all three in background
#   scripts/rv.sh stop         # graceful shutdown (SIGTERM then SIGKILL)
#   scripts/rv.sh status       # what's running / PIDs
#   scripts/rv.sh logs [name]  # tail -f (name: ds|api|frontend, default: all)
#   scripts/rv.sh restart

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── config (override via env) ─────────────────────────────────────────
CAMERAS_CFG="${RV_CAMERAS:-configs/cameras_live_ordered.json}"
MUX="${RV_MUX:-1280x720}"
RES_W="${RV_RES_W:-1280}"
RES_H="${RV_RES_H:-720}"
RES_CHANNEL="${RV_RES_CHANNEL:-102}"   # Hikvision substream
BITRATE="${RV_BITRATE:-8192}"          # kbps
RES_INTERVAL="${RV_RES_INTERVAL:-60}"  # seconds
ACTIVE_COLORS="${RV_ACTIVE_COLORS:-blue,green,red,yellow}"
FRONTEND_DIR="${RV_FRONTEND_DIR:-Kabirhan-Frontend}"
GO2RTC_BIN="${RV_GO2RTC_BIN:-bin/go2rtc}"
GO2RTC_CFG="${RV_GO2RTC_CFG:-configs/go2rtc_live.yaml}"

# ── video-only profile (file:// sources, no RTSP / go2rtc / Hikvision)
VIDEO_MODE=0
VIDEO_CAMERAS="${RV_VIDEO_CAMERAS:-configs/cameras_test_files_ordered.json}"
VIDEO_ACTIVE_COLORS="${RV_VIDEO_ACTIVE_COLORS:-red,green,yellow}"

# ── display profile (tiled X11 window with bbox/OSD; skips frontend)
DISPLAY_MODE=0
DISPLAY_W="${RV_DISPLAY_W:-1920}"
DISPLAY_H="${RV_DISPLAY_H:-1080}"

# ── paths ─────────────────────────────────────────────────────────────
PID_DIR="/tmp/rv_pids"
LOG_DIR="/tmp/rv_logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

svc_pid()  { cat "$PID_DIR/$1.pid" 2>/dev/null || echo ""; }
svc_log()  { echo "$LOG_DIR/$1.log"; }

# Match a running service by its command line (more reliable than PID
# files because setsid / npm / uvicorn tend to fork). Patterns avoid
# self-match by wrapping the first char in [...].
svc_match() {
    # ^-anchored so we don't match a bash shell whose cmdline just *contains*
    # these words (Claude's shell snapshots, "pkill -f ...", etc).
    case "$1" in
        go2rtc)   echo "go2rtc.*--config" ;;
        ds)       echo "^python[0-9.]* -m deepstream.main" ;;
        api)      echo "^python[0-9.]* -m api.server" ;;
        frontend) echo "^node .*vite" ;;
    esac
}
svc_pgrep() {
    local pat; pat=$(svc_match "$1")
    pgrep -f "$pat" 2>/dev/null | head -1
}
svc_alive(){
    local p; p=$(svc_pgrep "$1")
    if [[ -n "$p" ]]; then
        echo "$p" > "$PID_DIR/$1.pid"
        return 0
    fi
    rm -f "$PID_DIR/$1.pid"
    return 1
}

color_green() { printf "\e[32m%s\e[0m" "$1"; }
color_red()   { printf "\e[31m%s\e[0m" "$1"; }
color_dim()   { printf "\e[2m%s\e[0m"  "$1"; }

start_go2rtc() {
    if svc_alive go2rtc; then
        echo "  go2rtc  $(color_dim "already running (PID $(svc_pid go2rtc))")"
        return
    fi
    if [[ ! -x "$GO2RTC_BIN" ]]; then
        echo "  go2rtc  $(color_red "binary missing: $GO2RTC_BIN — skipped")"
        return
    fi
    if [[ ! -f "$GO2RTC_CFG" ]]; then
        echo "  go2rtc  $(color_red "config missing: $GO2RTC_CFG — skipped")"
        return
    fi
    setsid "$GO2RTC_BIN" --config "$GO2RTC_CFG" >"$(svc_log go2rtc)" 2>&1 &
    echo $! > "$PID_DIR/go2rtc.pid"
    sleep 2
    local real; real=$(svc_pgrep go2rtc); [[ -n "$real" ]] && echo "$real" > "$PID_DIR/go2rtc.pid"
    # Wait for API port 1984
    for _ in $(seq 1 10); do
        curl -sf -o /dev/null --max-time 1 http://localhost:1984/api/streams && { echo "  go2rtc  $(color_green started) PID=$(svc_pid go2rtc) log=$(svc_log go2rtc)"; echo "          API up (http://localhost:1984)"; return; }
        sleep 1
    done
    echo "  go2rtc  $(color_red 'API not responding at :1984 — check log')"
}

start_ds() {
    if svc_alive ds; then
        echo "  ds      $(color_dim "already running (PID $(svc_pid ds))")"
        return
    fi
    rm -f /dev/shm/rv_detections /dev/shm/sem.rv_detections_sem 2>/dev/null || true
    local ds_extra=""
    if [[ -n "${RV_DS_LIMIT:-}" ]]; then
        ds_extra="--limit $RV_DS_LIMIT"
    fi
    if [[ "$DISPLAY_MODE" -eq 1 ]]; then
        ds_extra="$ds_extra --display --display-width $DISPLAY_W --display-height $DISPLAY_H"
        export DISPLAY="${DISPLAY:-:0}"
    fi
    RV_ACTIVE_COLORS="$ACTIVE_COLORS" \
      setsid python3 -m deepstream.main \
        --cameras "$CAMERAS_CFG" $ds_extra \
        >"$(svc_log ds)" 2>&1 &
    echo $! > "$PID_DIR/ds.pid"
    echo "  ds      $(color_green started) PID=$! log=$(svc_log ds)"
    # wait up to 60s for SHM, so API has something to attach
    for _ in $(seq 1 60); do
        [[ -e /dev/shm/rv_detections ]] && break
        sleep 1
    done
    if [[ -e /dev/shm/rv_detections ]]; then
        echo "          SHM ready"
        local roi_line
        roi_line=$(grep -m1 "ROI loaded for" "$(svc_log ds)" 2>/dev/null || true)
        if [[ -n "$roi_line" ]]; then
            echo "          $(color_green 'ROI active') — $roi_line"
        else
            echo "          $(color_red 'ROI log line NOT found') — detections will NOT be polygon-filtered"
        fi
    else
        echo "          $(color_red 'SHM not created within 60s — check log')"
    fi
}

start_api() {
    if svc_alive api; then
        echo "  api     $(color_dim "already running (PID $(svc_pid api))")"
        return
    fi
    local api_extra=()
    # Resolution enforcement moved to standalone tools/set_camera_resolution.py
    # (main stream /101 1920x1080). api server no longer enforces anything.
    # Set RV_ENFORCE_INAPI=1 to re-enable the old in-api enforcer.
    if [[ "$VIDEO_MODE" -eq 0 ]] && [[ "${RV_ENFORCE_INAPI:-0}" -eq 1 ]]; then
        api_extra+=(--enforce-resolution "${RES_W}x${RES_H}"
                    --resolution-channel "$RES_CHANNEL"
                    --bitrate "$BITRATE"
                    --resolution-interval "$RES_INTERVAL")
    fi
    setsid python3 -m api.server \
        --config "$CAMERAS_CFG" \
        --deepstream --auto-start \
        "${api_extra[@]}" \
        >"$(svc_log api)" 2>&1 &
    echo $! > "$PID_DIR/api.pid"
    echo "  api     $(color_green started) PID=$! log=$(svc_log api)"
    # wait for HTTP to come up
    for _ in $(seq 1 30); do
        curl -sf -o /dev/null --max-time 1 http://localhost:8000/api/stats && { echo "          HTTP up"; return; }
        sleep 1
    done
    echo "          $(color_red 'HTTP not responding within 30s')"
}

start_frontend() {
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        echo "  frontend $(color_red "dir $FRONTEND_DIR missing — skipped")"
        return
    fi
    if svc_alive frontend; then
        echo "  frontend $(color_dim "already running (PID $(svc_pid frontend))")"
        return
    fi
    # node/npm are provided by nvm which isn't on the default PATH.
    # Source it before launching Vite. NVM_DIR can be overridden.
    local nvm_sh="${RV_NVM_SH:-/home/ipodrom/Рабочий стол/Ipodrom-Project/user/.nvm/nvm.sh}"
    if [[ ! -f "$nvm_sh" ]]; then
        echo "  frontend $(color_red "nvm.sh not found at $nvm_sh — set RV_NVM_SH")"
        return
    fi
    setsid bash -c "
        set -e
        export NVM_DIR=\"$(dirname "$nvm_sh")\"
        . '$nvm_sh'
        cd '$FRONTEND_DIR'
        exec npm run dev
    " >"$(svc_log frontend)" 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    sleep 2
    local real
    real=$(svc_pgrep frontend)
    [[ -n "$real" ]] && echo "$real" > "$PID_DIR/frontend.pid"
    echo "  frontend $(color_green started) PID=$(svc_pid frontend) log=$(svc_log frontend)"
}

stop_svc() {
    local name="$1"
    local pid; pid=$(svc_pid "$name")
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "  $name $(color_dim 'not running')"
        rm -f "$PID_DIR/$name.pid"
        return
    fi
    # kill whole process group (setsid above makes PID == PGID)
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        echo "  $name $(color_red 'force killed')"
    else
        echo "  $name $(color_green stopped)"
    fi
    rm -f "$PID_DIR/$name.pid"
}

cmd_start() {
    echo "==> starting Race Vision stack ($(date +%T))"
    echo "   cameras: $CAMERAS_CFG"
    if [[ "$VIDEO_MODE" -eq 1 ]]; then
        echo "   mode: $(color_green 'VIDEO (file://, no go2rtc, no Hikvision)')"
    else
        echo "   substream ch=$RES_CHANNEL ${RES_W}x${RES_H} @ ${BITRATE}kbps"
    fi
    if [[ "$DISPLAY_MODE" -eq 1 ]]; then
        echo "   display: $(color_green "tiled ${DISPLAY_W}x${DISPLAY_H} on DISPLAY=${DISPLAY:-:0}") (frontend skipped)"
    fi
    echo "   active colors: $ACTIVE_COLORS"
    if [[ "$VIDEO_MODE" -eq 0 ]]; then
        start_go2rtc
    else
        echo "  go2rtc  $(color_dim 'skipped (video mode)')"
    fi
    start_ds
    start_api
    if [[ "$DISPLAY_MODE" -eq 1 ]]; then
        echo "  frontend $(color_dim 'skipped (display mode)')"
    else
        start_frontend
    fi
    echo
    echo "  WebSocket : ws://localhost:8000/ws"
    if [[ "$DISPLAY_MODE" -eq 0 ]]; then
        echo "  Frontend  : http://localhost:5173"
    fi
    echo "  Stats     : curl http://localhost:8000/api/stats"
}

apply_video_mode() {
    VIDEO_MODE=1
    CAMERAS_CFG="$VIDEO_CAMERAS"
    ACTIVE_COLORS="$VIDEO_ACTIVE_COLORS"
    export RV_SNAP_MIN="${RV_SNAP_MIN:-0}"
}

apply_display_mode() {
    DISPLAY_MODE=1
}

parse_start_flags() {
    local f
    for f in "$@"; do
        case "$f" in
            --video)   apply_video_mode ;;
            --display) apply_display_mode ;;
            "") ;;
            *) echo "unknown flag: $f (allowed: --video, --display)"; exit 1 ;;
        esac
    done
}

cmd_stop() {
    echo "==> stopping Race Vision stack"
    # reverse order: frontend first, api, ds, go2rtc last
    stop_svc frontend
    stop_svc api
    stop_svc ds
    stop_svc go2rtc
    rm -f /dev/shm/rv_detections /dev/shm/sem.rv_detections_sem 2>/dev/null || true
}

cmd_status() {
    for s in go2rtc ds api frontend; do
        if svc_alive "$s"; then
            printf "  %-9s %s PID=%s\n" "$s" "$(color_green UP  )" "$(svc_pid "$s")"
        else
            printf "  %-9s %s\n" "$s" "$(color_red DOWN)"
        fi
    done
    printf "\n  SHM      "
    [[ -e /dev/shm/rv_detections ]] && color_green "EXISTS" || color_red "MISSING"
    echo
}

cmd_logs() {
    local target="${1:-all}"
    case "$target" in
        go2rtc|ds|api|frontend) exec tail -f -n 200 "$(svc_log "$target")" ;;
        all)                     exec tail -f -n 50 "$LOG_DIR"/*.log 2>/dev/null ;;
        *) echo "unknown log: $target (go2rtc|ds|api|frontend|all)"; exit 1 ;;
    esac
}

CMD="${1:-}"
shift || true

case "$CMD" in
    start)
        parse_start_flags "$@"
        cmd_start
        ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${1:-all}" ;;
    restart)
        parse_start_flags "$@"
        cmd_stop; sleep 1; cmd_start
        ;;
    "") echo "usage: $0 {start [--video] [--display]|stop|status|logs [ds|api|frontend]|restart [--video] [--display]}"; exit 1 ;;
    *)  echo "unknown: $CMD"; exit 1 ;;
esac
