#!/usr/bin/env bash
# Start / stop / restart the headless Isaac Sim 6.0 **Newton** kit with the
# Omniverse MCP bridge extension enabled.
#
# Why this exists: the Newton kit is normally launched by hand, and when its
# CUDA context faults (see NEWTON-GAIT-FINDINGS.md — constructing
# isaacsim.core.prims.SingleArticulation raises a sticky
# cudaErrorIllegalAddress that kills every later torch-CUDA call in the
# process) the only recovery is a full restart.  Reverse-engineering the
# command line out of /proc every time is slow, so it lives here.
#
# NEVER swap this for a PhysX experience to dodge a Newton bug — the whole
# point of this lane is Newton-stepped bodies.  The `physx` string that shows
# up when spawning a Go2 is the ASSET's variant-set name (it selects the
# rigid-body/joint payload), not the physics engine; the kit below is what
# chooses Newton.
#
# Usage:  ./newton_kit.sh {start|stop|restart|status} [port]
#
# Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0

set -uo pipefail

ISAAC_RELEASE="${ISAAC_RELEASE:-$HOME/Code/isaac-sim/IsaacSim/_build/linux-x86_64/release}"
MCP_EXT_FOLDER="${MCP_EXT_FOLDER:-$HOME/Code/omniverse-mcp/extension}"
KIT_APP="apps/isaacsim.exp.full.newton.kit"
PORT="${2:-8212}"
LOG="${NEWTON_KIT_LOG:-/tmp/newton_kit_${PORT}.log}"
PATTERN="isaacsim.exp.full.newton.kit.*port=${PORT}"

# The launch line matches several processes (the nohup wrapper shell as well
# as the kit binary itself), and killing the wrapper leaves the real kit --
# and its bridge socket -- very much alive.  Only accept a pid whose comm is
# literally "kit".
kit_pid() {
    local pid
    for pid in $(pgrep -f "$PATTERN"); do
        if [ "$(cat "/proc/$pid/comm" 2>/dev/null)" = "kit" ]; then
            echo "$pid"
            return 0
        fi
    done
    return 1
}

wait_for_bridge() {
    local tries="${1:-40}"
    for _ in $(seq 1 "$tries"); do
        if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
            echo "bridge listening on ${PORT}"
            return 0
        fi
        sleep 5
    done
    echo "TIMED OUT waiting for bridge on ${PORT}; tail of ${LOG}:" >&2
    tail -20 "$LOG" >&2
    return 1
}

case "${1:-status}" in
    start)
        if [ -n "$(kit_pid)" ]; then
            echo "already running (pid $(kit_pid))"
            exit 0
        fi
        [ -d "$ISAAC_RELEASE" ] || { echo "no Isaac release at $ISAAC_RELEASE" >&2; exit 1; }
        cd "$ISAAC_RELEASE" || exit 1
        nohup ./kit/kit "$KIT_APP" \
            --no-window \
            --ext-folder "$MCP_EXT_FOLDER" \
            --enable isaacsim.mcp.bridge \
            "--/exts/isaacsim.mcp.bridge/port=${PORT}" \
            > "$LOG" 2>&1 &
        echo "launched pid $! (log: $LOG)"
        wait_for_bridge
        ;;
    stop)
        pid="$(kit_pid)"
        if [ -z "$pid" ]; then echo "not running"; exit 0; fi
        kill "$pid"
        for _ in $(seq 1 12); do
            [ -z "$(kit_pid)" ] && { echo "stopped"; exit 0; }
            sleep 2
        done
        echo "did not exit on SIGTERM, sending SIGKILL" >&2
        kill -9 "$pid" 2>/dev/null
        ;;
    restart)
        "$0" stop "$PORT"
        sleep 3
        "$0" start "$PORT"
        ;;
    status)
        pid="$(kit_pid)"
        if [ -n "$pid" ]; then
            echo "running pid $pid"
        else
            echo "not running"
        fi
        ss -ltn 2>/dev/null | grep ":${PORT}\b" || echo "no listener on ${PORT}"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status} [port]" >&2
        exit 2
        ;;
esac
