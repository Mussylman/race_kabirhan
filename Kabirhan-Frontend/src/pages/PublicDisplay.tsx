import { useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence, useSpring, useTransform } from 'framer-motion';
import { Trophy } from 'lucide-react';
import { connectToBackend, disconnectFromBackend } from '../services/backendConnection';
import { useRaceStore } from '../store/raceStore';
import { useCameraStore } from '../store/cameraStore';
import { Go2RTCPlayer } from '../components/Go2RTCPlayer';
import { getSilkImagePath } from '../utils/silkUtils';

// Track position changes per horse for arc animations
const usePositionChanges = (rankings: { id: string; currentPosition: number }[]) => {
    const prevPositions = useRef<Record<string, number>>({});
    const [deltas, setDeltas] = useState<Record<string, number>>({});

    useEffect(() => {
        const newDeltas: Record<string, number> = {};
        let hasChange = false;

        for (const horse of rankings) {
            const prev = prevPositions.current[horse.id];
            if (prev !== undefined && prev !== horse.currentPosition) {
                newDeltas[horse.id] = horse.currentPosition - prev;
                hasChange = true;
            }
            prevPositions.current[horse.id] = horse.currentPosition;
        }

        if (hasChange) {
            setDeltas(newDeltas);
            const timer = setTimeout(() => setDeltas({}), 10000);
            return () => clearTimeout(timer);
        }
    }, [rankings]);

    return deltas;
};

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
    const { race, rankings } = useRaceStore();
    const { activePTZCameraId, ptzCameras, syncFromStorage } = useCameraStore();

    // Connect to backend WebSocket
    useEffect(() => {
        connectToBackend();
        return () => disconnectFromBackend();
    }, []);

    // Listen for camera changes from operator panel (other tabs)
    useEffect(() => {
        const handleStorageChange = () => syncFromStorage();
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, [syncFromStorage]);

    // Get active PTZ camera
    const activePTZ = ptzCameras.find(c => c.id === activePTZCameraId);

    const leader = rankings[0];
    const speed = leader ? leader.speed * 3.6 : 0;
    const time = leader?.timeElapsed || 0;
    const winner = rankings[0];

    // Top horses for ranking display
    const topHorses = rankings.slice(0, 10);

    // Track position changes for arc animation
    const positionDeltas = usePositionChanges(rankings);

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

            {/* BOTTOM - Professional TV Racing Bar */}
            <div className="absolute bottom-0 left-0 right-0 z-20">
                <div
                    className="border-t border-green-900/50"
                    style={{
                        background: 'linear-gradient(to right, #0a1a0a, #0d1f0d, #0a1a0a)'
                    }}
                >
                    <div className="flex items-center h-[140px]">

                        {/* LEFT - Position Marker */}
                        <div className="flex items-center justify-center w-24 h-full border-r border-green-900/30">
                            <div className="relative flex flex-col items-center">
                                <div className="w-12 h-12 rounded-full bg-red-600 border-4 border-white shadow-lg z-10"></div>
                                <div className="w-1.5 h-14 bg-gradient-to-b from-gray-300 to-gray-500 -mt-1"></div>
                            </div>
                        </div>

                        {/* CENTER - Jockeys with Numbers */}
                        <div className="flex-1 relative overflow-visible" style={{ minHeight: 140 }}>
                            {topHorses.map((horse, index) => {
                                const delta = positionDeltas[horse.id] || 0;
                                const isOvertake = delta < 0;
                                const isFallback = delta > 0;

                                const slotWidth = 90;
                                const containerCenter = 50;
                                const offsetPx = (index - (topHorses.length - 1) / 2) * slotWidth;

                                return (
                                    <motion.div
                                        key={horse.id}
                                        animate={{
                                            x: offsetPx,
                                            y: delta < 0 ? [-80, 0] : delta > 0 ? [50, 0] : 0,
                                            scale: delta !== 0 ? [1.3, 1] : 1,
                                        }}
                                        transition={{
                                            x: {
                                                type: 'tween',
                                                duration: 3,
                                                ease: [0.25, 0.1, 0.25, 1],
                                            },
                                            y: {
                                                type: 'tween',
                                                duration: 3,
                                                ease: [0.25, 0.1, 0.25, 1],
                                            },
                                            scale: {
                                                type: 'tween',
                                                duration: 2.5,
                                                ease: 'easeInOut',
                                            },
                                        }}
                                        className="absolute bottom-2 flex flex-col items-center w-[80px]"
                                        style={{
                                            left: `${containerCenter}%`,
                                            marginLeft: -40,
                                        }}
                                    >
                                        {/* Position change indicator arrow */}
                                        {isOvertake && (
                                            <motion.div
                                                className="absolute -top-8 left-1/2 -translate-x-1/2 text-green-400 font-bold text-lg z-10"
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: [0, 1, 1, 0], y: [10, -12, -12, -30] }}
                                                transition={{ duration: 10, times: [0, 0.05, 0.85, 1] }}
                                            >
                                                ▲ +{Math.abs(delta)}
                                            </motion.div>
                                        )}
                                        {isFallback && (
                                            <motion.div
                                                className="absolute -top-8 left-1/2 -translate-x-1/2 text-red-400 font-bold text-lg z-10"
                                                initial={{ opacity: 0, y: -10 }}
                                                animate={{ opacity: [0, 1, 1, 0], y: [-10, 0, 0, 15] }}
                                                transition={{ duration: 10, times: [0, 0.05, 0.85, 1] }}
                                            >
                                                ▼ -{Math.abs(delta)}
                                            </motion.div>
                                        )}

                                        {/* Jockey Icon with glow on change */}
                                        <img
                                            src={getSilkImagePath(horse.silkId)}
                                            alt={`#${horse.number}`}
                                            className="h-[70px] w-auto object-contain"
                                            style={{
                                                filter: isOvertake
                                                    ? 'drop-shadow(0 0 20px rgba(74, 222, 128, 0.9)) drop-shadow(0 0 40px rgba(74, 222, 128, 0.4)) drop-shadow(0 2px 4px rgba(0,0,0,0.5))'
                                                    : isFallback
                                                        ? 'drop-shadow(0 0 20px rgba(248, 113, 113, 0.8)) drop-shadow(0 0 40px rgba(248, 113, 113, 0.3)) drop-shadow(0 2px 4px rgba(0,0,0,0.5))'
                                                        : 'drop-shadow(0 2px 4px rgba(0,0,0,0.5))',
                                            }}
                                        />

                                        {/* Horse Number */}
                                        <div className={`font-bold text-xl mt-1 ${
                                            isOvertake ? 'text-green-300' : isFallback ? 'text-red-300' : 'text-white'
                                        }`}>
                                            {horse.number}
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </div>

                        {/* RIGHT - Speedometer */}
                        <div className="w-36 h-full flex items-center justify-center border-l border-green-900/30 bg-black/30">
                            <div className="text-center">
                                <div className="text-4xl font-bold text-white font-mono tabular-nums">
                                    <AnimatedNumber value={speed} decimals={1} />
                                </div>
                                <div className="text-sm text-gray-400 mt-1">km/h</div>
                            </div>
                        </div>

                    </div>
                </div>
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
