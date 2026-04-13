import { useState, useEffect, useRef, useCallback } from 'react';
import { Wifi, WifiOff, Loader2, EyeOff } from 'lucide-react';
import { GO2RTC_URL } from '../config/go2rtc';

interface Go2RTCPlayerProps {
    cameraId: string;
    cameraName?: string;
    className?: string;
    onVideoReady?: (video: HTMLVideoElement) => void;
    connectDelay?: number;
    lazy?: boolean;
}

export const Go2RTCPlayer = ({
    cameraId,
    cameraName = 'Camera',
    className = '',
    onVideoReady,
    lazy = false,
}: Go2RTCPlayerProps) => {
    const [status, setStatus] = useState<'connecting' | 'online' | 'offline' | 'paused'>(() => lazy ? 'paused' : 'connecting');
    const containerRef = useRef<HTMLDivElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const isVisibleRef = useRef(!lazy);
    const msRef = useRef<MediaSource | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const retriesRef = useRef(0);
    const mountedRef = useRef(true);

    const cleanup = useCallback(() => {
        if (reconnectTimerRef.current) {
            clearTimeout(reconnectTimerRef.current);
            reconnectTimerRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (msRef.current && msRef.current.readyState === 'open') {
            try { msRef.current.endOfStream(); } catch { /* ignore */ }
        }
        msRef.current = null;
        if (videoRef.current) {
            videoRef.current.src = '';
        }
    }, []);

    const connect = useCallback(() => {
        if (!mountedRef.current) return;
        if (!isVisibleRef.current) return;

        cleanup();
        setStatus('connecting');

        const ms = new MediaSource();
        msRef.current = ms;

        if (videoRef.current) {
            videoRef.current.src = URL.createObjectURL(ms);
        }

        ms.addEventListener('sourceopen', () => {
            if (!mountedRef.current) return;

            // Build WebSocket URL from go2rtc base URL
            const base = GO2RTC_URL.replace(/^http/, 'ws');
            const wsUrl = `${base}/api/ws?src=${encodeURIComponent(cameraId)}`;
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            let sourceBuffer: SourceBuffer | null = null;
            const queue: ArrayBuffer[] = [];
            let appending = false;

            const appendNext = () => {
                if (!sourceBuffer || sourceBuffer.updating || queue.length === 0) return;
                appending = true;
                const buf = queue.shift()!;
                try {
                    sourceBuffer.appendBuffer(buf);
                } catch {
                    // Buffer full or error — skip
                    appending = false;
                }
            };

            ws.onmessage = (event) => {
                if (!mountedRef.current) return;

                if (typeof event.data === 'string') {
                    // First message is the codec info (e.g. "video/mp4; codecs=\"avc1.640029\"")
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'mse' && msg.value) {
                        try {
                            sourceBuffer = ms.addSourceBuffer(msg.value);
                            sourceBuffer.mode = 'segments';
                            sourceBuffer.addEventListener('updateend', () => {
                                appending = false;
                                appendNext();
                            });
                        } catch {
                            setStatus('offline');
                            scheduleReconnect();
                        }
                    }
                } else {
                    // Binary data — mp4 segment
                    // Keep buffer small: drop old segments if queue grows
                    if (queue.length > 10) {
                        queue.splice(0, queue.length - 2);
                    }
                    queue.push(event.data);
                    if (!appending) appendNext();
                }
            };

            ws.onopen = () => {
                if (!mountedRef.current) return;
                setStatus('online');
                retriesRef.current = 0;
                // Send MSE request
                ws.send(JSON.stringify({ type: 'mse' }));
            };

            ws.onerror = () => {
                if (!mountedRef.current) return;
                setStatus('offline');
                scheduleReconnect();
            };

            ws.onclose = () => {
                if (!mountedRef.current) return;
                setStatus('offline');
                scheduleReconnect();
            };
        });
    }, [cameraId, cleanup]);

    // Track when video starts playing
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;
        const onPlaying = () => {
            setStatus('online');
            onVideoReady?.(video);
        };
        video.addEventListener('playing', onPlaying);
        return () => video.removeEventListener('playing', onPlaying);
    }, [onVideoReady]);

    const scheduleReconnect = useCallback(() => {
        if (!mountedRef.current) return;
        if (!isVisibleRef.current) return;
        if (reconnectTimerRef.current) return;
        const delay = Math.min(2000 * Math.pow(2, retriesRef.current), 30000);
        retriesRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            connect();
        }, delay);
    }, [connect]);

    // Lazy loading: connect/disconnect based on viewport visibility
    useEffect(() => {
        if (!lazy) return;
        const el = containerRef.current;
        if (!el) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                const visible = entry.isIntersecting;
                isVisibleRef.current = visible;
                if (visible) {
                    connect();
                } else {
                    cleanup();
                    setStatus('paused');
                    retriesRef.current = 0;
                }
            },
            { rootMargin: '100px' },
        );
        observer.observe(el);
        return () => observer.disconnect();
    }, [lazy, connect, cleanup]);

    useEffect(() => {
        mountedRef.current = true;
        if (!lazy) connect();
        return () => {
            mountedRef.current = false;
            cleanup();
        };
    }, [cameraId, connect, cleanup, lazy]);

    return (
        <div ref={containerRef} className={`relative bg-black overflow-hidden ${className}`}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />

            {status !== 'online' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {status === 'connecting' && (
                        <>
                            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-4" />
                            <p className="text-white text-sm">Connecting to stream...</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
                        </>
                    )}

                    {status === 'paused' && (
                        <>
                            <EyeOff className="w-12 h-12 text-gray-500 mb-4" />
                            <p className="text-white text-sm">Paused</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
                        </>
                    )}

                    {status === 'offline' && (
                        <>
                            <WifiOff className="w-12 h-12 text-red-400 mb-4" />
                            <p className="text-white text-sm">Stream Offline</p>
                            <p className="text-gray-500 text-xs mt-1">{cameraName}</p>
                        </>
                    )}
                </div>
            )}

            <div className="absolute top-2 left-2 flex items-center gap-2">
                <div className={`
                    flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium
                    ${status === 'online' ? 'bg-green-600' :
                        status === 'connecting' ? 'bg-blue-600' :
                        status === 'paused' ? 'bg-gray-600' : 'bg-red-600'}
                    text-white
                `}>
                    <Wifi className="w-3 h-3" />
                    {cameraName}
                </div>
            </div>
        </div>
    );
};
