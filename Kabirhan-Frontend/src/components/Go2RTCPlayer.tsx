import { useState, useEffect, useRef, useCallback } from 'react';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { GO2RTC_URL } from '../config/go2rtc';

interface Go2RTCPlayerProps {
    cameraId: string;
    cameraName?: string;
    className?: string;
    onVideoReady?: (video: HTMLVideoElement) => void;
}

export const Go2RTCPlayer = ({
    cameraId,
    cameraName = 'Camera',
    className = '',
    onVideoReady,
}: Go2RTCPlayerProps) => {
    const [status, setStatus] = useState<'connecting' | 'online' | 'offline'>('connecting');
    const videoRef = useRef<HTMLVideoElement>(null);
    const pcRef = useRef<RTCPeerConnection | null>(null);
    const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const mountedRef = useRef(true);

    const cleanup = useCallback(() => {
        if (reconnectTimerRef.current) {
            clearTimeout(reconnectTimerRef.current);
            reconnectTimerRef.current = null;
        }
        if (pcRef.current) {
            pcRef.current.close();
            pcRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    }, []);

    const connect = useCallback(async () => {
        if (!mountedRef.current) return;

        cleanup();
        setStatus('connecting');

        try {
            const pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
            });
            pcRef.current = pc;

            // Receive video track from go2rtc
            pc.addTransceiver('video', { direction: 'recvonly' });

            pc.ontrack = (event) => {
                if (!mountedRef.current) return;
                if (videoRef.current && event.streams[0]) {
                    videoRef.current.srcObject = event.streams[0];
                }
            };

            pc.oniceconnectionstatechange = () => {
                if (!mountedRef.current) return;
                const state = pc.iceConnectionState;
                if (state === 'connected' || state === 'completed') {
                    setStatus('online');
                    if (videoRef.current) {
                        onVideoReady?.(videoRef.current);
                    }
                } else if (state === 'disconnected' || state === 'failed' || state === 'closed') {
                    setStatus('offline');
                    scheduleReconnect();
                }
            };

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Send offer to go2rtc and get answer
            const res = await fetch(
                `${GO2RTC_URL}/api/webrtc?src=${encodeURIComponent(cameraId)}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/sdp' },
                    body: offer.sdp,
                }
            );

            if (!res.ok) {
                throw new Error(`go2rtc responded ${res.status}`);
            }

            const answerSdp = await res.text();
            await pc.setRemoteDescription(
                new RTCSessionDescription({ type: 'answer', sdp: answerSdp })
            );
        } catch {
            if (!mountedRef.current) return;
            setStatus('offline');
            scheduleReconnect();
        }
    }, [cameraId, cleanup, onVideoReady]);

    const scheduleReconnect = useCallback(() => {
        if (!mountedRef.current) return;
        if (reconnectTimerRef.current) return;
        reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            connect();
        }, 3000);
    }, [connect]);

    useEffect(() => {
        mountedRef.current = true;
        connect();
        return () => {
            mountedRef.current = false;
            cleanup();
        };
    }, [cameraId, connect, cleanup]);

    return (
        <div className={`relative bg-black overflow-hidden ${className}`}>
            {/* WebRTC Video */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
            />

            {/* Status Overlay */}
            {status !== 'online' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80">
                    {status === 'connecting' && (
                        <>
                            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-4" />
                            <p className="text-white text-sm">Connecting to stream...</p>
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

            {/* Camera Info Badge */}
            <div className="absolute top-2 left-2 flex items-center gap-2">
                <div className={`
                    flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium
                    ${status === 'online' ? 'bg-green-600' :
                        status === 'connecting' ? 'bg-blue-600' : 'bg-red-600'}
                    text-white
                `}>
                    <Wifi className="w-3 h-3" />
                    {cameraName}
                </div>
            </div>
        </div>
    );
};
