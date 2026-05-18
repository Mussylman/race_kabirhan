import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence, useSpring, useTransform } from 'framer-motion';
import { Trophy } from 'lucide-react';
import { connectToBackend, disconnectFromBackend } from '../services/backendConnection';
import { useRaceStore } from '../store/raceStore';
import { useCameraStore } from '../store/cameraStore';
import { Go2RTCPlayer } from '../components/Go2RTCPlayer';
import { getSilkImagePath } from '../utils/silkUtils';
import { RankingBoard } from '../components/public-display/RankingBoard';

// Format time as MM:SS.d
const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toFixed(1).padStart(4, '0')}`;
};

// Animated time component
const AnimatedTime = ({ value }: { value: number }) => {
    const spring = useSpring(value, { stiffness: 100, damping: 30 });
    const [displayValue, setDisplayValue] = useState(formatTime(value));

    useEffect(() => { spring.set(value); }, [spring, value]);
    useEffect(() => {
        const unsubscribe = spring.on('change', v => setDisplayValue(formatTime(v)));
        return unsubscribe;
    }, [spring]);

    return <span>{displayValue}</span>;
};

// Animated number component (for speed)
const AnimatedNumber = ({ value, decimals = 1 }: { value: number; decimals?: number }) => {
    const spring = useSpring(value, { stiffness: 100, damping: 30 });
    const display = useTransform(spring, v => v.toFixed(decimals));
    const [displayValue, setDisplayValue] = useState(value.toFixed(decimals));

    useEffect(() => { spring.set(value); }, [spring, value]);
    useEffect(() => {
        const unsubscribe = display.on('change', v => setDisplayValue(v));
        return unsubscribe;
    }, [display]);

    return <span>{displayValue}</span>;
};

export const PublicDisplay = () => {
    const { t } = useTranslation();
    const { race, rankings, initializeDefaultRace } = useRaceStore();
    const { activePTZCameraId, ptzCameras, syncFromStorage } = useCameraStore();

    // Initialize default horses + connect to backend
    useEffect(() => {
        initializeDefaultRace();
        connectToBackend();
        return () => disconnectFromBackend();
    }, []);

    // Demo mode DISABLED — show real backend data only, or static defaults.
    // (previous code swapped random adjacent positions every 5-8s)

    // Listen for camera changes from operator panel (other tabs)
    useEffect(() => {
        const handleStorageChange = () => syncFromStorage();
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, [syncFromStorage]);

    // Get active PTZ camera
    const activePTZ = ptzCameras.find(c => c.id === activePTZCameraId);

    const leader = rankings[0];
    const time = leader?.timeElapsed || 0;
    const winner = rankings[0];

    return (
        <div className="h-screen w-screen bg-black relative overflow-hidden">
            {/* PTZ Video Background via WebRTC */}
            {activePTZ && (
                <Go2RTCPlayer
                    cameraId={activePTZ.go2rtcId}
                    cameraName={activePTZ.name}
                    className="absolute inset-0 w-full h-full"
                />
            )}

            {/* TOP LEFT - Time & Lap */}
            <div className="absolute top-6 left-6 z-10">
                <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3 border border-white/10">
                    <div className="text-3xl font-bold text-white font-mono tabular-nums tracking-tight">
                        <AnimatedTime value={time} />
                    </div>
                    <div className="text-sm text-white/70 mt-1 font-medium">
                        {t('header.lap')} {leader?.currentLap || 1}/{race.totalLaps}
                    </div>
                </div>
            </div>

            {/* BOTTOM - Broadcast-style ranking bar */}
            <div className="absolute bottom-0 left-0 right-0 z-20">
                <RankingBoard rankings={rankings.slice(0, 4)} />
            </div>

            {/* Race Finished Overlay */}
            <AnimatePresence>
                {race.status === 'finished' && (
                    <motion.div
                        className="absolute inset-0 bg-black/90 flex items-center justify-center z-50"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.div
                            className="text-center"
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <Trophy className="w-20 h-20 text-amber-400 mx-auto mb-6" strokeWidth={1.5} />
                            <h2 className="text-4xl font-bold text-white mb-8">{t('display.raceFinished')}</h2>

                            {winner && (
                                <div className="flex items-center justify-center gap-6">
                                    <div className="w-28 h-36 flex items-center justify-center">
                                        <img
                                            src={getSilkImagePath(winner.silkId)}
                                            alt={`Winner silk`}
                                            className="w-24 h-32 object-contain drop-shadow-[0_8px_16px_rgba(0,0,0,0.5)]"
                                        />
                                    </div>
                                    <div className="text-left">
                                        <div className="text-xs text-amber-400 uppercase tracking-wider mb-1">{t('display.winner')}</div>
                                        <p className="text-2xl font-bold text-white">{winner.name}</p>
                                        <p className="text-gray-400">{winner.jockeyName}</p>
                                        <p className="text-amber-400 font-mono text-xl mt-2">#{winner.number}</p>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
