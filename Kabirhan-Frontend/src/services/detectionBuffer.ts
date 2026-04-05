/**
 * DetectionBuffer — per-camera temporal buffer for time-aware bbox rendering.
 *
 * WebSocket pushes detection frames here (not directly to render).
 * rAF render loop queries the buffer for the frame closest to video.currentTime.
 */

import { flog, throttledLog, frontendAgg } from '../utils/frameLogger';

export interface Detection {
    color: string;
    conf: number;
    track_id?: number;
    bbox?: [number, number, number, number];
}

export interface DetectionFrame {
    ts_capture: number;      // seconds — capture time from C++ SHM
    ts_server_send: number;  // seconds — when server sent the WS message
    ts_client_recv: number;  // seconds — when browser received it
    frame_seq: number;       // SHM write_seq — cross-layer tracing key
    frame_w: number;
    frame_h: number;
    detections: Detection[];
}

interface CameraBuffer {
    frames: DetectionFrame[];
}

// --- Buffer storage ---

const BUFFER_MAX_AGE_S = 2.0;   // keep last 2 seconds
const STALE_THRESHOLD_MS = 300;  // clear overlay if no fresh data

const buffers = new Map<string, CameraBuffer>();

function getBuffer(camId: string): CameraBuffer {
    let buf = buffers.get(camId);
    if (!buf) {
        buf = { frames: [] };
        buffers.set(camId, buf);
    }
    return buf;
}

/**
 * Push a new detection frame into the buffer.
 * Called from WebSocket message handler.
 */
export function pushDetectionFrame(
    camId: string,
    camData: { frame_w: number; frame_h: number; ts_capture?: number; frame_seq?: number; detections: Detection[] },
    tsServerSend: number,
): void {
    const tsClientRecv = Date.now() / 1000;
    const frameSeq = camData.frame_seq ?? 0;
    const frame: DetectionFrame = {
        ts_capture: camData.ts_capture ?? 0,
        ts_server_send: tsServerSend,
        ts_client_recv: tsClientRecv,
        frame_seq: frameSeq,
        frame_w: camData.frame_w,
        frame_h: camData.frame_h,
        detections: camData.detections,
    };

    const buf = getBuffer(camId);
    const sizeBefore = buf.frames.length;
    buf.frames.push(frame);

    // Evict old frames
    const cutoff = tsClientRecv - BUFFER_MAX_AGE_S;
    while (buf.frames.length > 0 && buf.frames[0].ts_client_recv < cutoff) {
        buf.frames.shift();
    }
    const evicted = sizeBefore - (buf.frames.length - 1);

    // Record in aggregator
    const e2eMs = (tsClientRecv - frame.ts_capture) * 1000;
    frontendAgg.recordWs(camId, e2eMs);

    // BUFFER_PUSH log — throttled 2s per camera
    throttledLog(`BUFFER_PUSH:${camId}`, 2000, () => {
        flog('BUFFER_PUSH', camId, frameSeq, tsClientRecv, {
            buf_size: buf.frames.length,
            evicted: evicted > 0 ? evicted : undefined,
            dets: frame.detections.length,
        });
    });
}

export type FrameSelectResult =
    | { status: 'DRAW'; frame: DetectionFrame; delta_ms: number }
    | { status: 'STALE'; reason: string }
    | { status: 'NO_FRAME' };

/**
 * Select the best detection frame for rendering.
 *
 * @param camId       camera ID
 * @param videoTime   video.currentTime (seconds, media clock)
 * @param syncOffset  manual offset in seconds (positive = metadata arrives later than video)
 */
export function selectFrame(
    camId: string,
    videoTime: number,
    syncOffset: number = 0,
): FrameSelectResult {
    const buf = buffers.get(camId);

    if (!buf || buf.frames.length === 0) {
        frontendAgg.recordNoFrame(camId);
        throttledLog(`FRAME_SELECT_NOFRAME:${camId}`, 500, () => {
            flog('FRAME_SELECT', camId, '?', Date.now() / 1000, { video_t: videoTime, status: 'NO_FRAME' });
        });
        frontendAgg.flushIfDue();
        return { status: 'NO_FRAME' };
    }

    // Target time: video clock adjusted by sync offset
    const targetT = videoTime + syncOffset;

    // Find frame with ts_capture closest to targetT
    let bestIdx = 0;
    let bestDelta = Math.abs(buf.frames[0].ts_capture - targetT);
    for (let i = 1; i < buf.frames.length; i++) {
        const delta = Math.abs(buf.frames[i].ts_capture - targetT);
        if (delta < bestDelta) {
            bestDelta = delta;
            bestIdx = i;
        }
    }

    const best = buf.frames[bestIdx];
    const deltaMsAbs = bestDelta * 1000;

    // Check staleness
    const nowS = Date.now() / 1000;
    const ageSinceRecv = (nowS - best.ts_client_recv) * 1000;
    if (ageSinceRecv > STALE_THRESHOLD_MS) {
        frontendAgg.recordStale(camId);
        throttledLog(`FRAME_SELECT_STALE:${camId}`, 500, () => {
            flog('FRAME_SELECT', camId, best.frame_seq, nowS, {
                video_t: videoTime, delta_ms: deltaMsAbs, age_ms: ageSinceRecv, status: 'STALE',
            });
        });
        frontendAgg.flushIfDue();
        return { status: 'STALE', reason: `age=${ageSinceRecv.toFixed(0)}ms` };
    }

    frontendAgg.recordDraw(camId, deltaMsAbs);
    throttledLog(`FRAME_SELECT_DRAW:${camId}`, 2000, () => {
        flog('FRAME_SELECT', camId, best.frame_seq, nowS, {
            video_t: videoTime, delta_ms: deltaMsAbs, status: 'DRAW',
        });
    });
    frontendAgg.flushIfDue();

    return { status: 'DRAW', frame: best, delta_ms: deltaMsAbs };
}

/**
 * Get the latest frame for a camera (fallback when video.currentTime is unavailable).
 */
export function getLatestFrame(camId: string): DetectionFrame | null {
    const buf = buffers.get(camId);
    if (!buf || buf.frames.length === 0) return null;
    return buf.frames[buf.frames.length - 1];
}

/**
 * Clear buffer for a camera.
 */
export function clearBuffer(camId: string): void {
    buffers.delete(camId);
    flog('CLEAR', camId, 0, Date.now() / 1000, { reason: 'manual' });
}
