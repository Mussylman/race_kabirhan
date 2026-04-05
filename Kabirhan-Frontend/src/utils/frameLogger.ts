/**
 * frameLogger.ts — Structured logging helpers for Race Vision frontend.
 *
 * Provides:
 *   flog(stage, camId, frameSeq, ts, fields)
 *       Emits one structured log line to console.debug (filtered by LOG flags).
 *
 *   throttle(key, intervalMs, fn)
 *       Calls fn() only if enough time has passed since last call for that key.
 *
 *   frontendAgg
 *       Per-camera stats aggregator — call record*() freely, emits STATS_1S once/second.
 *
 * Log flags — set via .env (Vite) or at runtime in DevTools:
 *   window.__LOG_FLAGS.LOG_WS     = true  → WS_RECV logs
 *   window.__LOG_FLAGS.LOG_BUFFER = true  → BUFFER_PUSH logs
 *   window.__LOG_FLAGS.LOG_VIDEO  = true  → FRAME_SELECT, DRAW, CLEAR logs
 *   window.__LOG_FLAGS.LOG_TIMING = true  → disables throttle for timing stages
 *   window.__LOG_FLAGS.LOG_STATS  = true  → STATS_1S logs (default on)
 *
 * Vite build-time defaults:
 *   VITE_LOG_WS, VITE_LOG_BUFFER, VITE_LOG_VIDEO, VITE_LOG_TIMING, VITE_LOG_STATS
 */

// ── Flag resolution ────────────────────────────────────────────────────

declare global {
    interface Window {
        __LOG_FLAGS?: Partial<LogFlags>;
    }
}

interface LogFlags {
    LOG_WS:     boolean;
    LOG_BUFFER: boolean;
    LOG_VIDEO:  boolean;
    LOG_TIMING: boolean;
    LOG_STATS:  boolean;
}

function getFlag(name: keyof LogFlags, envVar: string, defaultVal: boolean): boolean {
    // Runtime override takes priority
    const rt = window.__LOG_FLAGS?.[name];
    if (rt !== undefined) return rt;
    // Vite env
    const env = (import.meta.env as Record<string, string>)[envVar];
    if (env !== undefined) return env === 'true';
    return defaultVal;
}

/** Live-read flags so DevTools overrides take effect immediately. */
export function flags(): LogFlags {
    return {
        LOG_WS:     getFlag('LOG_WS',     'VITE_LOG_WS',     false),
        LOG_BUFFER: getFlag('LOG_BUFFER', 'VITE_LOG_BUFFER', false),
        LOG_VIDEO:  getFlag('LOG_VIDEO',  'VITE_LOG_VIDEO',  false),
        LOG_TIMING: getFlag('LOG_TIMING', 'VITE_LOG_TIMING', false),
        LOG_STATS:  getFlag('LOG_STATS',  'VITE_LOG_STATS',  true),
    };
}

// ── Stage → flag mapping ───────────────────────────────────────────────

type Stage =
    | 'WS_RECV'
    | 'BUFFER_PUSH'
    | 'FRAME_SELECT'
    | 'DRAW'
    | 'CLEAR'
    | 'STATS_1S';

function stageEnabled(stage: Stage): boolean {
    const f = flags();
    switch (stage) {
        case 'WS_RECV':                     return f.LOG_WS;
        case 'BUFFER_PUSH':                 return f.LOG_BUFFER;
        case 'FRAME_SELECT':
        case 'DRAW':
        case 'CLEAR':                       return f.LOG_VIDEO;
        case 'STATS_1S':                    return f.LOG_STATS;
    }
}

// ── Structured log emitter ─────────────────────────────────────────────

type Fields = Record<string, string | number | boolean | undefined>;

/**
 * Emit one structured log line.
 *
 * Example output (console.debug):
 *   WS_RECV        cam=cam-05  frame=14823  ts=1743417600.198  dets=3  latency_ms=55.0
 */
export function flog(
    stage: Stage,
    camId: string,
    frameSeq: number | string,
    ts: number,
    fields: Fields = {},
): void {
    if (!stageEnabled(stage)) return;

    const parts: string[] = [
        `cam=${camId}`,
        `frame=${frameSeq}`,
        `ts=${ts.toFixed(3)}`,
    ];
    for (const [k, v] of Object.entries(fields)) {
        if (v === undefined) continue;
        if (typeof v === 'number') {
            parts.push(`${k}=${Number.isInteger(v) ? v : v.toFixed(1)}`);
        } else {
            parts.push(`${k}=${v}`);
        }
    }

    console.debug(`%-14s  ${parts.join('  ')}`, stage);
}

// ── Throttle ───────────────────────────────────────────────────────────

const _throttleMap = new Map<string, number>();

/**
 * Call fn() only if intervalMs has passed since the last call for key.
 * Returns true if fn was called.
 */
export function throttledLog(
    key: string,
    intervalMs: number,
    fn: () => void,
): boolean {
    // When LOG_TIMING is on, bypass throttle for timing-sensitive stages
    if (flags().LOG_TIMING) {
        fn();
        return true;
    }
    const now = performance.now();
    const last = _throttleMap.get(key) ?? -Infinity;
    if (now - last >= intervalMs) {
        _throttleMap.set(key, now);
        fn();
        return true;
    }
    return false;
}

// ── PerCameraAggregator ────────────────────────────────────────────────

interface CamWindow {
    windowStart: number;   // performance.now() ms
    wsCount:     number;
    drawCount:   number;
    deltaSum:    number;
    deltaMax:    number;
    staleCount:  number;
    noFrameCount: number;
    e2eSum:      number;   // ts_client_recv - ts_capture (ms)
    e2eCount:    number;
}

const _aggWindows = new Map<string, CamWindow>();

function getWindow(camId: string): CamWindow {
    let w = _aggWindows.get(camId);
    if (!w) {
        w = {
            windowStart: performance.now(),
            wsCount: 0, drawCount: 0,
            deltaSum: 0, deltaMax: 0,
            staleCount: 0, noFrameCount: 0,
            e2eSum: 0, e2eCount: 0,
        };
        _aggWindows.set(camId, w);
    }
    return w;
}

export const frontendAgg = {
    recordWs(camId: string, e2eMs: number): void {
        const w = getWindow(camId);
        w.wsCount++;
        w.e2eSum += e2eMs;
        w.e2eCount++;
    },
    recordDraw(camId: string, deltaMs: number): void {
        const w = getWindow(camId);
        w.drawCount++;
        w.deltaSum += deltaMs;
        w.deltaMax = Math.max(w.deltaMax, deltaMs);
    },
    recordStale(camId: string): void {
        getWindow(camId).staleCount++;
    },
    recordNoFrame(camId: string): void {
        getWindow(camId).noFrameCount++;
    },
    /** Call once per rAF or once per second to flush completed windows. */
    flushIfDue(): void {
        if (!flags().LOG_STATS) return;
        const now = performance.now();
        for (const [camId, w] of _aggWindows) {
            const elapsed = now - w.windowStart;
            if (elapsed < 1000) continue;

            const elapsedS = elapsed / 1000;
            const wsFps    = w.wsCount    / elapsedS;
            const drawFps  = w.drawCount  / elapsedS;
            const avgDelta = w.drawCount  > 0 ? w.deltaSum  / w.drawCount  : 0;
            const e2eAvg   = w.e2eCount   > 0 ? w.e2eSum    / w.e2eCount   : 0;
            const total    = w.drawCount + w.staleCount + w.noFrameCount;
            const stalePct = total > 0 ? (w.staleCount   / total * 100) : 0;
            const buf      = 0; // caller can pass if needed

            const ts = Date.now() / 1000;
            flog('STATS_1S', camId, '—', ts, {
                ws_fps:        wsFps,
                draw_fps:      drawFps,
                avg_delta_ms:  avgDelta,
                max_delta_ms:  w.deltaMax,
                stale_pct:     stalePct,
                e2e_avg_ms:    e2eAvg,
                stale:         w.staleCount,
                no_frame:      w.noFrameCount,
            });

            // Reset
            _aggWindows.set(camId, {
                windowStart: now,
                wsCount: 0, drawCount: 0,
                deltaSum: 0, deltaMax: 0,
                staleCount: 0, noFrameCount: 0,
                e2eSum: 0, e2eCount: 0,
            });
        }
    },
};
