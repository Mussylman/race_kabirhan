import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { getSilkImagePath } from '../../utils/silkUtils';

// ── Types ──────────────────────────────────────────────────────────────

interface Horse {
    id: string;
    number?: number;
    name?: string;
    jockeyName?: string;
    silkId: number;
    currentPosition: number;
    lastCameraId?: string;
    color?: string;
}

interface Props {
    rankings: Horse[];
}

// ── Design tokens ──────────────────────────────────────────────────────

const POSITION_COLOR: Record<number, { strip: string; rgba: string; label: string }> = {
    1: { strip: '#FFB800', rgba: 'rgba(255, 184, 0, 0.55)',  label: 'P1' },
    2: { strip: '#C0C7D1', rgba: 'rgba(192, 199, 209, 0.45)', label: 'P2' },
    3: { strip: '#CD7F32', rgba: 'rgba(205, 127, 50, 0.45)',  label: 'P3' },
    4: { strip: '#6B7280', rgba: 'rgba(107, 114, 128, 0.40)', label: 'P4' },
};

const FONT_DISPLAY = '"Inter", system-ui, sans-serif';
const FONT_MONO    = '"JetBrains Mono", "SF Mono", monospace';

// ── Sparkline (15 sec position history) ────────────────────────────────

const SPARK_W = 56;
const SPARK_H = 10;
const SPARK_MAX_POINTS = 15;

const Sparkline = ({ history, color }: { history: number[]; color: string }) => {
    if (history.length < 2) {
        return <div style={{ width: SPARK_W, height: SPARK_H }} />;
    }
    // Lower position number = better → draw "up" on the sparkline.
    // We invert so y is bigger for worse positions.
    const min = Math.min(...history);
    const max = Math.max(...history);
    const span = Math.max(1, max - min);
    const step = SPARK_W / (history.length - 1);
    const points = history
        .map((pos, i) => {
            const x = i * step;
            // pos is 1-based; smaller is better → smaller y (top of svg).
            const y = ((pos - min) / span) * (SPARK_H - 2) + 1;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');
    return (
        <svg
            width={SPARK_W}
            height={SPARK_H}
            viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
            style={{ display: 'block' }}
        >
            <motion.polyline
                fill="none"
                stroke={color}
                strokeWidth={1.25}
                strokeLinecap="round"
                strokeLinejoin="round"
                points={points}
                initial={{ pathLength: 0.85, opacity: 0.6 }}
                animate={{ pathLength: 1, opacity: 0.95 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
            />
        </svg>
    );
};

// ── Per-horse position history hook ────────────────────────────────────

function usePositionHistory(rankings: Horse[]): Record<string, number[]> {
    const [history, setHistory] = useState<Record<string, number[]>>({});
    const prevRef = useRef<Record<string, number>>({});

    useEffect(() => {
        const next: Record<string, number[]> = { ...history };
        let changed = false;
        for (const h of rankings) {
            const prev = prevRef.current[h.id];
            if (prev !== h.currentPosition) {
                const arr = (next[h.id] || []).concat(h.currentPosition);
                next[h.id] = arr.slice(-SPARK_MAX_POINTS);
                prevRef.current[h.id] = h.currentPosition;
                changed = true;
            } else if (!(h.id in next)) {
                next[h.id] = [h.currentPosition];
                prevRef.current[h.id] = h.currentPosition;
                changed = true;
            }
        }
        if (changed) setHistory(next);
    }, [rankings]);

    return history;
}

// ── Position change tracker (for LED-flash overtake animation) ─────────

function usePositionDeltas(rankings: Horse[]): Record<string, number> {
    const [deltas, setDeltas] = useState<Record<string, number>>({});
    const prevRef = useRef<Record<string, number>>({});

    useEffect(() => {
        const next: Record<string, number> = {};
        let changed = false;
        for (const h of rankings) {
            const prev = prevRef.current[h.id];
            if (prev !== undefined && prev !== h.currentPosition) {
                next[h.id] = h.currentPosition - prev;
                changed = true;
            }
            prevRef.current[h.id] = h.currentPosition;
        }
        if (changed) {
            setDeltas(next);
            const t = setTimeout(() => setDeltas({}), 1500);
            return () => clearTimeout(t);
        }
    }, [rankings]);

    return deltas;
}

// ── Jockey card ────────────────────────────────────────────────────────

const JockeyCard = ({
    horse,
    history,
    delta,
    isLeader,
}: {
    horse: Horse;
    history: number[];
    delta: number;
    isLeader: boolean;
}) => {
    const pc = POSITION_COLOR[horse.currentPosition] || POSITION_COLOR[4];
    const isOvertake = delta < 0;
    const isFallback = delta > 0;
    // Leader breathes 3x amplitude; others very subtle.
    const breathAmp = isLeader ? 1.015 : 1.005;

    return (
        <motion.div
            layout
            className="relative flex-1 flex items-stretch h-full overflow-hidden"
            transition={{ layout: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] } }}
        >
            {/* Position color strip */}
            <div
                className="w-1.5 h-full"
                style={{ background: pc.strip, boxShadow: isLeader ? `0 0 12px ${pc.strip}` : undefined }}
            />

            {/* Card body */}
            <div className="flex-1 flex items-center gap-4 px-4 relative">
                {/* Position label */}
                <div className="flex flex-col items-start min-w-[44px]">
                    <span
                        style={{
                            fontFamily: FONT_MONO,
                            fontWeight: 700,
                            fontSize: 28,
                            color: pc.strip,
                            lineHeight: 1,
                            letterSpacing: '-0.02em',
                        }}
                    >
                        {pc.label}
                    </span>
                </div>

                {/* Silk image with breathing animation */}
                <motion.img
                    src={getSilkImagePath(horse.silkId)}
                    alt={`#${horse.number ?? '?'}`}
                    className="object-contain"
                    style={{ height: 64, width: 'auto' }}
                    animate={{ scale: [1, breathAmp, 1] }}
                    transition={{
                        duration: 4,
                        ease: 'easeInOut',
                        repeat: Infinity,
                    }}
                />

                {/* Number + name + sparkline */}
                <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                        <span
                            style={{
                                fontFamily: FONT_MONO,
                                fontWeight: 700,
                                fontSize: 18,
                                color: pc.strip,
                                lineHeight: 1,
                            }}
                        >
                            #{horse.number ?? '?'}
                        </span>
                        <span
                            style={{
                                fontFamily: FONT_DISPLAY,
                                fontWeight: 600,
                                fontSize: 18,
                                color: '#F5F7FA',
                                letterSpacing: '0.04em',
                                textTransform: 'uppercase',
                                lineHeight: 1,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                            }}
                        >
                            {horse.jockeyName || horse.name || '—'}
                        </span>
                    </div>
                    <Sparkline history={history || []} color={pc.strip} />
                    {/* TODO: dynamic status when lap/gap/track-position data available */}
                </div>
            </div>

            {/* LED-flash overtake indicator (top→bottom sweep) */}
            <AnimatePresence>
                {isOvertake && (
                    <motion.div
                        key="overtake"
                        className="absolute left-0 right-0 pointer-events-none"
                        style={{
                            height: 10,
                            background:
                                'linear-gradient(180deg, transparent 0%, #00FF88 50%, transparent 100%)',
                            boxShadow: '0 0 14px rgba(0, 255, 136, 0.9)',
                        }}
                        initial={{ top: 0, opacity: 0 }}
                        animate={{ top: '100%', opacity: [0, 1, 1, 0] }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                    />
                )}
                {isFallback && (
                    <motion.div
                        key="fallback"
                        className="absolute left-0 right-0 pointer-events-none"
                        style={{
                            height: 10,
                            background:
                                'linear-gradient(180deg, transparent 0%, #FF3B3B 50%, transparent 100%)',
                            boxShadow: '0 0 14px rgba(255, 59, 59, 0.9)',
                        }}
                        initial={{ bottom: 0, opacity: 0 }}
                        animate={{ bottom: '100%', opacity: [0, 1, 1, 0] }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                    />
                )}
            </AnimatePresence>
        </motion.div>
    );
};

// ── Leader callout (right side of bar) ─────────────────────────────────

const LeaderCallout = ({ leader }: { leader: Horse | undefined }) => {
    if (!leader) {
        return <div className="w-[240px] border-l border-white/10" />;
    }
    return (
        <div
            className="w-[240px] flex flex-col items-center justify-center gap-1 border-l border-white/10 px-4"
            style={{ background: 'rgba(255, 184, 0, 0.04)' }}
        >
            <span
                style={{
                    fontFamily: FONT_DISPLAY,
                    fontWeight: 600,
                    fontSize: 11,
                    color: '#FFB800',
                    letterSpacing: '0.18em',
                    textTransform: 'uppercase',
                    opacity: 0.85,
                }}
            >
                ◆ Leader
            </span>
            <AnimatePresence mode="wait">
                <motion.div
                    key={leader.id}
                    className="flex items-center gap-3"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.35, ease: 'easeOut' }}
                >
                    <motion.div className="relative" style={{ height: 96 }}>
                        {/* Gold halo (expansion on leader change) */}
                        <motion.div
                            className="absolute inset-0 rounded-full pointer-events-none"
                            style={{
                                boxShadow: '0 0 24px rgba(255, 184, 0, 0.55)',
                            }}
                            initial={{ scale: 0.4, opacity: 0 }}
                            animate={{ scale: 1.1, opacity: [0, 0.9, 0] }}
                            transition={{ duration: 0.6, ease: 'easeOut' }}
                        />
                        <motion.img
                            src={getSilkImagePath(leader.silkId)}
                            alt={`#${leader.number ?? '?'}`}
                            className="object-contain relative"
                            style={{
                                height: 96,
                                width: 'auto',
                                filter: 'drop-shadow(0 0 18px rgba(255, 184, 0, 0.45))',
                            }}
                            animate={{ scale: [1, 1.015, 1] }}
                            transition={{ duration: 4, ease: 'easeInOut', repeat: Infinity }}
                        />
                    </motion.div>
                    <div className="flex flex-col gap-0.5">
                        <span
                            style={{
                                fontFamily: FONT_MONO,
                                fontWeight: 700,
                                fontSize: 22,
                                color: '#FFB800',
                                lineHeight: 1,
                            }}
                        >
                            #{leader.number ?? '?'}
                        </span>
                        <span
                            style={{
                                fontFamily: FONT_DISPLAY,
                                fontWeight: 700,
                                fontSize: 18,
                                color: '#F5F7FA',
                                letterSpacing: '0.05em',
                                textTransform: 'uppercase',
                                lineHeight: 1.1,
                            }}
                        >
                            {leader.jockeyName || leader.name || '—'}
                        </span>
                        {/* Subtle gold accent line + idle pulse */}
                        <motion.div
                            className="mt-1"
                            style={{
                                height: 2,
                                width: 60,
                                background: 'linear-gradient(90deg, #FFB800 0%, rgba(255,184,0,0) 100%)',
                            }}
                            animate={{ opacity: [0.4, 1, 0.4] }}
                            transition={{ duration: 2, ease: 'easeInOut', repeat: Infinity }}
                        />
                    </div>
                </motion.div>
            </AnimatePresence>
        </div>
    );
};

// ── Main bar ───────────────────────────────────────────────────────────

export const RankingBoard = ({ rankings }: Props) => {
    const top4 = rankings.slice(0, 4);
    const sorted = [...top4].sort((a, b) => a.currentPosition - b.currentPosition);
    const leader = sorted[0];
    const history = usePositionHistory(top4);
    const deltas = usePositionDeltas(top4);

    return (
        <div
            className="w-full h-[120px] flex items-stretch border-t border-white/10"
            style={{
                background:
                    'linear-gradient(to right, #0A1628, #0E1A30, #0A1628)',
            }}
        >
            <LayoutGroup id="ranking-board">
                <div className="flex-1 flex items-stretch">
                    {sorted.map((horse) => (
                        <JockeyCard
                            key={horse.id}
                            horse={horse}
                            history={history[horse.id] || []}
                            delta={deltas[horse.id] || 0}
                            isLeader={horse.id === leader?.id}
                        />
                    ))}
                </div>
            </LayoutGroup>
            <LeaderCallout leader={leader} />
        </div>
    );
};
