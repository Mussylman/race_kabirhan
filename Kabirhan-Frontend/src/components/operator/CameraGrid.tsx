import { Video, VideoOff } from 'lucide-react';
import { useCameraStore } from '../../store/cameraStore';
import { useRaceStore } from '../../store/raceStore';
import { Go2RTCPlayer } from '../Go2RTCPlayer';
import { TRACK_LENGTH } from '../../types';

export const CameraGrid = () => {
    const { analyticsCameras } = useCameraStore();
    const { rankings } = useRaceStore();

    return (
        <div className="p-6 h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="mb-4 flex-shrink-0">
                <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">Analytics Cameras</h2>
                <p className="text-sm text-[var(--text-muted)]">
                    {analyticsCameras.length} cameras — YOLO + CNN detection (not shown to viewers)
                </p>
            </div>

            {/* Camera Grid — 3 cameras side by side */}
            <div className="grid grid-cols-3 gap-4 flex-1 overflow-y-auto">
                {analyticsCameras.map((camera) => {
                    const isOnline = camera.status === 'online';

                    // Find horses within this camera's track segment
                    const horsesNear = rankings.filter(horse => {
                        const horsePos = horse.distanceCovered % TRACK_LENGTH;
                        return horsePos >= camera.trackStart && horsePos < camera.trackEnd;
                    });
                    const hasHorses = horsesNear.length > 0;

                    return (
                        <div
                            key={camera.id}
                            className={`
                                relative rounded-xl overflow-hidden transition-all duration-300
                                ${hasHorses
                                    ? 'ring-2 ring-amber-500 shadow-lg shadow-amber-500/20'
                                    : 'border border-[var(--border)]'}
                            `}
                        >
                            {/* WebRTC Stream via go2rtc */}
                            <div className="aspect-video bg-black">
                                <Go2RTCPlayer
                                    cameraId={camera.go2rtcId}
                                    cameraName={camera.name}
                                    className="w-full h-full"
                                />
                            </div>

                            {/* Bottom Info Bar */}
                            <div className="p-3 bg-[var(--surface)]">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        {isOnline ? (
                                            <Video className={`w-4 h-4 ${hasHorses ? 'text-amber-400' : 'text-[var(--text-muted)]'}`} />
                                        ) : (
                                            <VideoOff className="w-4 h-4 text-[var(--danger)]" />
                                        )}
                                        <span className={`text-sm font-semibold ${hasHorses ? 'text-amber-400' : 'text-[var(--text-primary)]'}`}>
                                            {camera.name}
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-500'}`} />
                                        <span className="text-xs text-[var(--text-muted)]">
                                            {camera.trackStart}–{camera.trackEnd}m
                                        </span>
                                    </div>
                                </div>

                                {/* Horse detection badge */}
                                {hasHorses && (
                                    <div className="mt-2 flex items-center gap-1">
                                        <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full font-medium">
                                            {horsesNear.length} horse{horsesNear.length > 1 ? 's' : ''} detected
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
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                    <span>Offline</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded border-2 border-amber-500" />
                    <span>Horse Detected</span>
                </div>
            </div>
        </div>
    );
};
