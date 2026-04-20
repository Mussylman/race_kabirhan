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
ACTIVE_COLORS="${RV_ACTIVE_COLORS:-blue,green,purple,red,yellow}"
FRONTEND_DIR="${RV_FRONTEND_DIR:-Kabirhan-Frontend}"
GO2RTC_BIN="${RV_GO2RTC_BIN:-bin/go2rtc}"
GO2RTC_CFG="${RV_GO2RTC_CFG:-configs/go2rtc_live.yaml}"

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
    else
        echo "          $(color_red 'SHM not created within 60s — check log')"
    fi
}

start_api() {
    if svc_alive api; then
        echo "  api     $(color_dim "already running (PID $(svc_pid api))")"
        return
    fi
    setsid python3 -m api.server \
        --config "$CAMERAS_CFG" \
        --deepstream --auto-start \
        --enforce-resolution "${RES_W}x${RES_H}" \
        --resolution-channel "$RES_CHANNEL" \
        --bitrate "$BITRATE" \
        --resolution-interval "$RES_INTERVAL" \
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
    echo "   substream ch=$RES_CHANNEL ${RES_W}x${RES_H} @ ${BITRATE}kbps"
    echo "   active colors: $ACTIVE_COLORS"
    start_go2rtc
    start_ds
    start_api
    start_frontend
    echo
    echo "  WebSocket : ws://localhost:8000/ws"
    echo "  Frontend  : http://localhost:5173"
    echo "  Stats     : curl http://localhost:8000/api/stats"
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

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${2:-all}" ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    "") echo "usage: $0 {start|stop|status|logs [ds|api|frontend]|restart}"; exit 1 ;;
    *)  echo "unknown: $1"; exit 1 ;;
esac
