import { useRef, useEffect, useCallback } from 'react';
import { Camera, Wifi, Maximize } from 'lucide-react';
import { useCameraStore } from '../../store/cameraStore';
import { useRaceStore } from '../../store/raceStore';
import { Go2RTCPlayer } from '../Go2RTCPlayer';
import { SILK_COLORS } from '../../types';

export const PTZCameraDisplay = () => {
    const { ptzCameras, activePTZCameraId } = useCameraStore();
    const { race, rankings } = useRaceStore();
    const activeCamera = ptzCameras.find(cam => cam.id === activePTZCameraId);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const animFrameRef = useRef<number>(0);

    const handleVideoReady = useCallback((video: HTMLVideoElement) => {
        videoRef.current = video;
    }, []);

    // Canvas overlay: draw horse markers from WebSocket ranking data
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const draw = () => {
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            // Match canvas size to its display size
            const rect = canvas.getBoundingClientRect();
            if (canvas.width !== rect.width || canvas.height !== rect.height) {
                canvas.width = rect.width;
                canvas.height = rect.height;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (rankings.length === 0) {
                animFrameRef.current = requestAnimationFrame(draw);
                return;
            }

            // Draw horse labels in bottom-right area above camera selector
            const labelX = canvas.width - 20;
            let labelY = canvas.height - 80;

            for (let i = Math.min(rankings.length, 5) - 1; i >= 0; i--) {
                const horse = rankings[i];
                const color = SILK_COLORS[horse.silkId] || horse.color || '#fff';
                const speed = (horse.speed * 3.6).toFixed(1);

                // Background pill
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                const text = `#${horse.number}  ${speed} km/h`;
                ctx.font = 'bold 14px system-ui, sans-serif';
                const metrics = ctx.measureText(text);
                const pillW = metrics.width + 30;
                const pillH = 24;
                const pillX = labelX - pillW;
                const pillY = labelY - pillH / 2;

                ctx.beginPath();
                ctx.roundRect(pillX, pillY, pillW, pillH, 4);
                ctx.fill();

                // Color dot
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(pillX + 12, labelY, 5, 0, Math.PI * 2);
                ctx.fill();

                // Text
                ctx.fillStyle = '#fff';
                ctx.fillText(text, pillX + 22, labelY + 5);

                labelY -= 30;
            }

            animFrameRef.current = requestAnimationFrame(draw);
        };

        animFrameRef.current = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(animFrameRef.current);
    }, [rankings]);

    return (
        <div className="relative w-full h-full bg-[var(--background)]">
            {/* Camera Feed via WebRTC */}
            {activeCamera ? (
                <div className="absolute inset-0">
                    <Go2RTCPlayer
                        cameraId={activeCamera.go2rtcId}
                        cameraName={activeCamera.name}
                        className="w-full h-full"
                        onVideoReady={handleVideoReady}
                    />
                    {/* Canvas overlay for horse markers */}
                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full pointer-events-none"
                    />
                </div>
            ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                        <Camera className="w-16 h-16 text-[var(--border)] mx-auto mb-4" strokeWidth={1} />
                        <p className="text-sm text-[var(--text-muted)]">No Camera Feed</p>
                    </div>
                </div>
            )}

            {/* Top Left */}
            <div className="absolute top-4 left-4 flex items-center gap-3 z-10">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--surface)] rounded-lg border border-[var(--border)]">
                    <Camera className="w-4 h-4 text-[var(--text-muted)]" />
                    <span className="text-sm font-medium text-[var(--text-primary)]">
                        {activeCamera?.name || 'PTZ 1'}
                    </span>
                </div>

                <div className="live-badge">LIVE</div>
            </div>

            {/* Top Right */}
            <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-[var(--surface)] rounded-lg border border-[var(--border)]">
                    <Wifi className="w-3.5 h-3.5 text-[var(--accent)]" />
                    <span className="text-xs text-[var(--text-secondary)]">HD</span>
                </div>

                <button className="p-1.5 bg-[var(--surface)] rounded-lg border border-[var(--border)] hover:border-[var(--text-muted)] transition-colors cursor-pointer">
                    <Maximize className="w-4 h-4 text-[var(--text-muted)]" />
                </button>
            </div>

            {/* Bottom Left */}
            <div className="absolute bottom-4 left-4 z-10">
                <div className="px-4 py-2.5 bg-[var(--surface)] rounded-lg border border-[var(--border)]">
                    <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Race</p>
                    <p className="text-sm font-medium text-[var(--text-primary)]">
                        {race.name || 'Grand Championship'}
                    </p>
                </div>
            </div>

            {/* Bottom Right - Camera Selector */}
            <div className="absolute bottom-4 right-4 flex gap-1 z-10">
                {ptzCameras.map((cam, i) => (
                    <div
                        key={cam.id}
                        className={`
              w-8 h-6 rounded flex items-center justify-center text-xs font-medium cursor-pointer transition-colors
              ${cam.id === activePTZCameraId
                                ? 'bg-[var(--primary)] text-white'
                                : 'bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:text-[var(--text-secondary)]'}
            `}
                    >
                        {i + 1}
                    </div>
                ))}
            </div>
        </div>
    );
};
