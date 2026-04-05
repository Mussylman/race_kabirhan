import { useRef, useEffect, useCallback, useState } from 'react';
import { Video, VideoOff } from 'lucide-react';
import { useCameraStore } from '../../store/cameraStore';
import { useRaceStore } from '../../store/raceStore';
import { GO2RTC_URL } from '../../config/go2rtc';
import { TRACK_LENGTH } from '../../types';
import { getLatestFrame, type DetectionFrame } from '../../services/detectionBuffer';
import { flog, throttledLog } from '../../utils/frameLogger';

/**
 * MSE video stream from go2rtc.
 * Protocol: client sends {type:'mse', value:codecs} → server sends {type:'mse', value:codec} → binary segments.
 */
const VideoStream = ({ cameraId, videoRef: externalRef }: { cameraId: string; videoRef?: React.RefObject<HTMLVideoElement | null> }) => {
    const internalRef = useRef<HTMLVideoElement>(null);
    const videoRef = externalRef ?? internalRef;

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const wsUrl = GO2RTC_URL.replace(/^http/, 'ws') + `/api/ws?src=${cameraId}`;
        let ws: WebSocket | null = null;
        let ms: MediaSource | null = null;
        let sb: SourceBuffer | null = null;
        let buf = new Uint8Array(2 * 1024 * 1024); // 2MB pre-allocated
        let bufLen = 0;
        let running = true;

        // Build supported codecs string
        const codecs = (isSupported: (codec: string) => boolean): string => {
            const dominated = [
                'video/mp4; codecs="avc1.640029"',  // H.264 High 4.1
                'video/mp4; codecs="avc1.64001f"',  // H.264 High 3.1
                'video/mp4; codecs="avc1.42e01e"',  // H.264 Baseline 3.0
                'video/mp4; codecs="hvc1.1.6.L153"', // H.265
            ];
            return dominated.filter(c => isSupported(c)).join(', ');
        };

        const connect = () => {
            if (!running) return;

            ms = new MediaSource();
            ms.addEventListener('sourceopen', () => {
                // Tell go2rtc what codecs we support
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    ws!.send(JSON.stringify({
                        type: 'mse',
                        value: codecs(MediaSource.isTypeSupported),
                    }));
                };

                ws.onmessage = (ev) => {
                    if (typeof ev.data === 'string') {
                        const msg = JSON.parse(ev.data);
                        if (msg.type === 'mse' && ms!.readyState === 'open') {
                            // Server responds with actual codec to use
                            sb = ms!.addSourceBuffer(msg.value);
                            sb.mode = 'segments';
                            sb.addEventListener('updateend', () => {
                                if (!sb!.updating && bufLen > 0) {
                                    try {
                                        sb!.appendBuffer(buf.slice(0, bufLen));
                                        bufLen = 0;
                                    } catch { /* ignore */ }
                                }
                                // Keep buffer trimmed to last 5s
                                if (!sb!.updating && sb!.buffered.length) {
                                    const end = sb!.buffered.end(sb!.buffered.length - 1);
                                    const start0 = sb!.buffered.start(0);
                                    if (end - start0 > 5) {
                                        sb!.remove(start0, end - 5);
                                    }
                                    // Chase live edge
                                    if (video.currentTime < end - 1) {
                                        video.currentTime = end - 0.5;
                                    }
                                }
                            });
                        }
                        return;
                    }

                    // Binary H.264 segment
                    if (!sb) return;
                    if (sb.updating || bufLen > 0) {
                        // Queue into pre-allocated buffer
                        const data = new Uint8Array(ev.data);
                        if (bufLen + data.length > buf.length) {
                            // Grow buffer
                            const newBuf = new Uint8Array(buf.length * 2);
                            newBuf.set(buf.subarray(0, bufLen));
                            buf = newBuf;
                        }
                        buf.set(data, bufLen);
                        bufLen += data.length;
                    } else {
                        try {
                            sb.appendBuffer(ev.data);
                        } catch { /* ignore */ }
                    }
                };

                ws.onclose = () => { if (running) setTimeout(connect, 2000); };
                ws.onerror = () => ws?.close();
            }, { once: true });

            video.src = URL.createObjectURL(ms);
            video.play().catch(() => {});
        };

        connect();

        return () => {
            running = false;
            ws?.close();
            if (ms?.readyState === 'open') {
                try { ms.endOfStream(); } catch { /* ignore */ }
            }
        };
    }, [cameraId]);

    return (
        <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="absolute inset-0 w-full h-full object-contain"
        />
    );
};

const COLOR_MAP: Record<string, string> = {
    blue: '#3b82f6',
    green: '#22c55e',
    purple: '#a855f7',
    red: '#ef4444',
    yellow: '#eab308',
    unknown: '#6b7280',
};

/**
 * Time-aware canvas overlay — draws bbox from DetectionBuffer, not directly from store.
 *
 * Render loop via rAF:
 *   1. Get video.currentTime (or fallback to latest frame)
 *   2. Apply syncOffset
 *   3. Find nearest detection frame by ts_capture
 *   4. Draw or clear if stale
 */
const SYNC_OFFSET = 0; // seconds — reserved for future temporal alignment
const STALE_THRESHOLD_MS = 1000; // ms — clear overlay if no fresh detection data

const DetectionOverlay = ({ cameraId, videoRef }: { cameraId: string; videoRef?: React.RefObject<HTMLVideoElement | null> }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rafRef = useRef<number>(0);

    useEffect(() => {
        let running = true;

        const renderLoop = () => {
            if (!running) return;

            const canvas = canvasRef.current;
            if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    const rect = canvas.getBoundingClientRect();
                    canvas.width = rect.width;
                    canvas.height = rect.height;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Use latest frame with staleness check.
                    // Note: full temporal alignment (video.currentTime vs ts_capture) requires
                    // a calibration offset and is deferred — the ts_capture is unix epoch while
                    // currentTime is media time, so we can't directly compare them.
                    let frame: DetectionFrame | null = getLatestFrame(cameraId);
                    if (frame) {
                        const age = (Date.now() / 1000 - frame.ts_client_recv) * 1000;
                        if (age > STALE_THRESHOLD_MS) {
                            throttledLog(`CLEAR:${cameraId}`, 500, () => {
                                flog('CLEAR', cameraId, frame!.frame_seq, Date.now() / 1000, {
                                    reason: 'STALE', age_ms: age,
                                });
                            });
                            frame = null;
                        }
                    }

                    if (frame && frame.detections.length > 0) {
                        // Bbox coords are in mux space (e.g. 1520x1520 square).
                        // Video is 16:9 letterboxed inside the square mux with maintain-aspect-ratio.
                        // We need to undo the letterbox padding before scaling to canvas.
                        const muxW = frame.frame_w;
                        const muxH = frame.frame_h;
                        const videoAspect = canvas.width / canvas.height; // display aspect (16:9)
                        const muxAspect = muxW / muxH;
                        let padX = 0, padY = 0, activeW = muxW, activeH = muxH;
                        if (muxAspect > videoAspect) {
                            // pillarbox — black bars on sides
                            activeW = muxH * videoAspect;
                            padX = (muxW - activeW) / 2;
                        } else {
                            // letterbox — black bars top/bottom
                            activeH = muxW / videoAspect;
                            padY = (muxH - activeH) / 2;
                        }
                        const scaleX = canvas.width / activeW;
                        const scaleY = canvas.height / activeH;

                        for (const det of frame.detections) {
                            if (!det.bbox) continue;
                            const [x1, y1, x2, y2] = det.bbox;
                            const sx = (x1 - padX) * scaleX;
                            const sy = (y1 - padY) * scaleY;
                            const sw = (x2 - x1) * scaleX;
                            const sh = (y2 - y1) * scaleY;

                            const color = COLOR_MAP[det.color] || COLOR_MAP.unknown;

                            // Draw bbox
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(sx, sy, sw, sh);

                            // Draw label background
                            const label = `${det.color.toUpperCase()} ${det.conf}%`;
                            ctx.font = 'bold 11px sans-serif';
                            const textW = ctx.measureText(label).width + 8;
                            ctx.fillStyle = color;
                            ctx.fillRect(sx, sy - 18, textW, 18);

                            // Draw label text
                            ctx.fillStyle = '#fff';
                            ctx.fillText(label, sx + 4, sy - 5);
                        }

                        // DRAW log — throttled 5s per camera (rAF is 60fps, very noisy)
                        throttledLog(`DRAW:${cameraId}`, 5000, () => {
                            const ts = Date.now() / 1000;
                            const boxes = frame!.detections
                                .map(d => `${d.color}:${d.conf}%`)
                                .join(' ');
                            flog('DRAW', cameraId, frame!.frame_seq, ts, {
                                dets: frame!.detections.length,
                                boxes,
                            });
                        });
                    } else if (!frame) {
                        // CLEAR log — throttled 500ms per camera
                        throttledLog(`CLEAR:${cameraId}`, 500, () => {
                            const ts = Date.now() / 1000;
                            flog('CLEAR', cameraId, 0, ts, { reason: 'NO_FRAME' });
                        });
                    }
                }
            }

            rafRef.current = requestAnimationFrame(renderLoop);
        };

        rafRef.current = requestAnimationFrame(renderLoop);
        return () => {
            running = false;
            cancelAnimationFrame(rafRef.current);
        };
    }, [cameraId, videoRef]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 z-10 pointer-events-none"
        />
    );
};

/** Wires VideoStream and DetectionOverlay to the same videoRef for time-aware bbox rendering */
const CameraCell = ({ cameraId, go2rtcId }: { cameraId: string; go2rtcId: string }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    return (
        <div className="relative aspect-video bg-black">
            <VideoStream cameraId={go2rtcId} videoRef={videoRef} />
            <DetectionOverlay cameraId={cameraId} videoRef={videoRef} />
        </div>
    );
};

// All camera IDs — show all that have go2rtc streams
const ACTIVE_CAMERA_IDS = new Set([
    'cam-01','cam-02','cam-03','cam-04','cam-05',
    'cam-06','cam-07','cam-08','cam-09','cam-10',
    'cam-11','cam-12','cam-13','cam-14','cam-15',
    'cam-16','cam-17','cam-18','cam-19','cam-20',
    'cam-21','cam-22','cam-23','cam-24','cam-25',
]);

export const CameraGrid = () => {
    const { analyticsCameras, liveDetections } = useCameraStore();
    const { rankings } = useRaceStore();

    // Show only cameras that have go2rtc streams
    const visibleCameras = analyticsCameras.filter(c => ACTIVE_CAMERA_IDS.has(c.id));

    return (
        <div className="p-6 h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="mb-4 flex-shrink-0">
                <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">Analytics Cameras</h2>
                <p className="text-sm text-[var(--text-muted)]">
                    {visibleCameras.length} cameras — YOLO + CNN detection
                </p>
            </div>

            {/* Camera Grid — 5 cameras per row for 25 total */}
            <div className="grid grid-cols-5 gap-2 flex-1 overflow-y-auto">
                {visibleCameras.map((camera) => {
                    const isOnline = camera.status === 'online';

                    // Find horses within this camera's track segment (from rankings)
                    const horsesNear = rankings.filter(horse => {
                        const horsePos = horse.distanceCovered % TRACK_LENGTH;
                        return horsePos >= camera.trackStart && horsePos < camera.trackEnd;
                    });
                    const hasHorses = horsesNear.length > 0;

                    // Live detection from DeepStream C++ (real-time, with colors)
                    const camData = liveDetections[camera.id];
                    const dets = camData?.detections || [];
                    const detCount = dets.length;
                    const isActive = detCount > 0;

                    return (
                        <div
                            key={camera.id}
                            onDoubleClick={(e) => e.currentTarget.requestFullscreen()}
                            className={`
                                relative rounded-xl overflow-hidden transition-all duration-300
                                ${isActive
                                    ? 'ring-2 ring-green-500 shadow-lg shadow-green-500/30'
                                    : hasHorses
                                        ? 'ring-2 ring-amber-500 shadow-lg shadow-amber-500/20'
                                        : 'border border-[var(--border)]'}
                            `}
                        >
                            {/* ACTIVE badge — top-right overlay */}
                            {isActive && (
                                <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 bg-green-500 text-white text-xs font-bold px-2.5 py-1 rounded-full shadow-lg animate-pulse">
                                    <div className="w-2 h-2 rounded-full bg-white" />
                                    ACTIVE
                                </div>
                            )}

                            {/* Video Stream via go2rtc MSE + Detection overlay */}
                            <CameraCell cameraId={camera.id} go2rtcId={camera.go2rtcId} />

                            {/* Bottom Info Bar */}
                            <div className="p-3 bg-[var(--surface)]">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        {isOnline ? (
                                            <Video className={`w-4 h-4 ${isActive ? 'text-green-400' : hasHorses ? 'text-amber-400' : 'text-[var(--text-muted)]'}`} />
                                        ) : (
                                            <VideoOff className="w-4 h-4 text-[var(--danger)]" />
                                        )}
                                        <span className={`text-sm font-semibold ${isActive ? 'text-green-400' : hasHorses ? 'text-amber-400' : 'text-[var(--text-primary)]'}`}>
                                            {camera.name}
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : isOnline ? 'bg-emerald-500' : 'bg-red-500'}`} />
                                        <span className="text-xs text-[var(--text-muted)]">
                                            {camera.trackStart}–{camera.trackEnd}m
                                        </span>
                                    </div>
                                </div>

                                {/* Detection info with colors */}
                                {isActive && Array.isArray(dets) && (
                                    <div className="mt-2 flex flex-wrap items-center gap-1">
                                        {dets.map((d, i) => {
                                            const colorMap: Record<string, string> = {
                                                blue: 'bg-blue-500',
                                                green: 'bg-green-500',
                                                purple: 'bg-purple-500',
                                                red: 'bg-red-500',
                                                yellow: 'bg-yellow-400',
                                            };
                                            const bgClass = colorMap[d.color] || 'bg-gray-500';
                                            return (
                                                <span key={i} className={`text-xs text-white px-2 py-0.5 rounded-full font-medium flex items-center gap-1 ${bgClass}`}>
                                                    <span className="capitalize">{d.color}</span>
                                                    <span className="opacity-75">{d.conf}%</span>
                                                    {d.track_id ? <span className="opacity-60">#{d.track_id}</span> : null}
                                                </span>
                                            );
                                        })}
                                    </div>
                                )}
                                {!isActive && hasHorses && (
                                    <div className="mt-2 flex items-center gap-1">
                                        <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full font-medium">
                                            {horsesNear.length} horse{horsesNear.length > 1 ? 's' : ''} in segment
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center justify-center gap-8 text-xs text-[var(--text-muted)] flex-shrink-0 pt-3 border-t border-[var(--border)]">
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                    <span>Online</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
                    <span>Active (detecting)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded border-2 border-amber-500" />
                    <span>Horse in Segment</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                    <span>Offline</span>
                </div>
            </div>
        </div>
    );
};
