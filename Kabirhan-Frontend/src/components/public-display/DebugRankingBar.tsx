/**
 * DebugRankingBar — TEMPORARY visual diagnostic for ranking pipeline.
 *
 * Two rows side-by-side, both reading the same useRaceStore.rankings:
 *   BACKEND   — instant snapshot, no animation (just .map of horses in store order)
 *   ANIMATION — same data wrapped in framer-motion LayoutGroup + layoutId
 *
 * When backend pushes a new ranking, BACKEND row replaces immediately
 * (one render); ANIMATION row plays a layout-spring re-order. Visually
 * comparing the two reveals whether animations lag, freeze, or skip
 * intermediate states.
 *
 * Keyboard: press 'D' to toggle visibility (default ON).
 *
 * REMOVE BEFORE PRODUCTION (search for: [DEBUG-RANKING-BAR]).
 */

import { useEffect, useState } from 'react';
import { motion, LayoutGroup } from 'framer-motion';
import { useRaceStore } from '../../store/raceStore';

const COLOR_MAP: Record<string, string> = {
    blue:   '#2563EB',
    green:  '#22C55E',
    red:    '#DC2626',
    yellow: '#EAB308',
};

const formatTime = (ms: number): string => {
    const d = new Date(ms);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
};

export const DebugRankingBar = () => {
    const { rankings } = useRaceStore();
    const [visible, setVisible] = useState(true);
    const [lastUpdateMs, setLastUpdateMs] = useState<number>(Date.now());
    const [lastSig, setLastSig] = useState<string>('');

    // Detect ranking change → bump timestamp
    useEffect(() => {
        const sig = rankings
            .map(h => `${h.id}:${h.currentPosition}:${h.lastCameraId || ''}`)
            .join('|');
        if (sig !== lastSig) {
            setLastSig(sig);
            setLastUpdateMs(Date.now());
        }
    }, [rankings, lastSig]);

    // Keyboard toggle ('d' / 'D')
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement) return;
            if (e.key === 'd' || e.key === 'D') {
                setVisible(v => !v);
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    if (!visible) return null;

    // Source order — exactly as store gives us (which is sorted by currentPosition).
    // We render position labels 1..N from index, and use horse.lastCameraId for context.
    const horses = rankings.slice(0, 10);
    const lastCam = horses[0]?.lastCameraId || '?';

    const Circle = ({ horse, index }: { horse: typeof horses[number]; index: number }) => {
        const bg = COLOR_MAP[horse.color] || '#6B7280';
        return (
            <div className="flex flex-col items-center">
                <div
                    style={{ background: bg, borderColor: '#fff' }}
                    className="w-10 h-10 rounded-full border-2 flex items-center justify-center text-white font-bold text-sm shadow-md"
                >
                    {index + 1}
                </div>
                <div className="text-[9px] text-white/70 mt-0.5 font-mono">
                    {horse.color || `#${horse.number}`}
                </div>
            </div>
        );
    };

    return (
        <div
            style={{
                position: 'fixed',
                top: 10,
                right: 10,
                zIndex: 9999,
                background: 'rgba(0, 0, 0, 0.85)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: 8,
                padding: '8px 12px',
                color: 'white',
                fontFamily: 'system-ui, -apple-system, sans-serif',
                fontSize: 11,
                minWidth: 460,
                pointerEvents: 'none', // don't block click-through
            }}
        >
            <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-amber-300 text-[10px] tracking-wider">
                    DEBUG RANKING BAR
                </span>
                <span className="text-[9px] text-white/50 font-mono">press D to toggle</span>
            </div>

            {/* BACKEND row — instant snapshot, no animation */}
            <div className="flex items-center gap-3 mb-2">
                <div className="text-[10px] font-bold text-cyan-300 w-20 flex-shrink-0">
                    BACKEND
                </div>
                <div className="flex gap-2">
                    {horses.length === 0 ? (
                        <span className="text-white/40 text-[10px]">no rankings</span>
                    ) : (
                        horses.map((h, i) => (
                            <Circle key={`bg-${h.id}`} horse={h} index={i} />
                        ))
                    )}
                </div>
            </div>

            {/* ANIMATION row — same data, animated via LayoutGroup */}
            <div className="flex items-center gap-3 mb-1">
                <div className="text-[10px] font-bold text-emerald-300 w-20 flex-shrink-0">
                    ANIMATION
                </div>
                <LayoutGroup>
                    <div className="flex gap-2">
                        {horses.length === 0 ? (
                            <span className="text-white/40 text-[10px]">no rankings</span>
                        ) : (
                            horses.map((h, i) => (
                                <motion.div
                                    key={`anim-${h.id}`}
                                    layout
                                    layoutId={`debug-${h.id}`}
                                    transition={{ type: 'spring', stiffness: 300, damping: 28 }}
                                >
                                    <Circle horse={h} index={i} />
                                </motion.div>
                            ))
                        )}
                    </div>
                </LayoutGroup>
            </div>

            <div className="flex items-center justify-between text-[10px] text-white/60 font-mono mt-1 pt-1 border-t border-white/10">
                <span>cam: <span className="text-white">{lastCam}</span></span>
                <span>last update: <span className="text-white">{formatTime(lastUpdateMs)}</span></span>
                <span>{horses.length} horses</span>
            </div>
        </div>
    );
};
