import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { Eye, ChevronUp, ChevronDown } from 'lucide-react';
import { getSilkImagePath } from '../../utils/silkUtils';
import { useRaceStore } from '../../store/raceStore';
import { useCameraStore } from '../../store/cameraStore';

// ── Types ──────────────────────────────────────────────────────────────

interface Horse {
    id: string;
    number?: number;
    name?: string;
    jockeyName?: string;
    silkId: number;
    currentPosition: number;
    currentLap?: number;
    lastCameraId?: string;
    color?: string;
}

interface Props {
    rankings: Horse[];
}

// ── Design tokens (light editorial broadcast theme) ────────────────────

const COLOR_BG       = 'rgba(255, 255, 255, 0.96)';
const COLOR_STRIP    = '#F5F5F7';   // Apple-style neutral
const COLOR_FG       = '#0A0A0A';   // deeper black, premium
const COLOR_MUTED    = '#71717A';   // Zinc-500
const COLOR_LEADER   = '#C77800';   // amber gold, less olive
const COLOR_UP       = '#10B981';   // Emerald-500
const COLOR_DOWN     = '#EF4444';   // Red-500
const COLOR_DIVIDER  = 'rgba(0, 0, 0, 0.08)';
const COLOR_BORDER   = 'rgba(0, 0, 0, 0.06)';  // hairline for light-on-light contrast

const FONT_DISPLAY = '"Geist", system-ui, sans-serif';
const FONT_MONO    = '"Geist Mono", "SF Mono", monospace';

// ── Position-change deltas (3 sec arrow indicator) ─────────────────────

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
            const t = setTimeout(() => setDeltas({}), 3000);
            return () => clearTimeout(t);
        }
    }, [rankings]);

    return deltas;
}

// ── Jockey cell ────────────────────────────────────────────────────────

const JockeyCell = ({
    horse,
    isLeader,
    delta,
    isLast,
}: {
    horse: Horse;
    isLeader: boolean;
    delta: number;
    isLast: boolean;
}) => {
    const isUp = delta < 0;
    const isDown = delta > 0;

    return (
        <motion.div
            layout
            className="relative flex-1 flex items-center gap-3 h-full"
            style={{ paddingLeft: 20, paddingRight: 20 }}
            transition={{ layout: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] } }}
        >
            {/* Circular silk */}
            <div
                className="flex-shrink-0 rounded-full overflow-hidden"
                style={{
                    width: 56,
                    height: 56,
                    background: 'rgba(0,0,0,0.04)',
                    boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.08)',
                }}
            >
                <img
                    src={getSilkImagePath(horse.silkId)}
                    alt={`#${horse.number ?? '?'}`}
                    className="w-full h-full object-cover"
                />
            </div>

            {/* Name + (optional) LEADER label */}
            <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                {isLeader && (
                    <span
                        style={{
                            fontFamily: FONT_DISPLAY,
                            fontSize: 10,
                            fontWeight: 600,
                            color: COLOR_LEADER,
                            letterSpacing: '0.22em',
                            textTransform: 'uppercase',
                            lineHeight: 1,
                        }}
                    >
                        Leader
                    </span>
                )}
                <span
                    style={{
                        fontFamily: FONT_DISPLAY,
                        fontSize: 20,
                        fontWeight: 600,
                        color: COLOR_FG,
                        letterSpacing: '0.04em',
                        textTransform: 'uppercase',
                        lineHeight: 1.05,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                    }}
                >
                    {horse.jockeyName || horse.name || '—'}
                </span>
            </div>

            {/* Position change arrow */}
            <AnimatePresence>
                {(isUp || isDown) && (
                    <motion.div
                        key={isUp ? 'up' : 'down'}
                        className="flex-shrink-0 flex items-center justify-center"
                        style={{ width: 28, height: 28 }}
                        initial={{ opacity: 0, x: 8, scale: 0.85 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: -8, scale: 0.85 }}
                        transition={{ duration: 0.3, ease: 'easeOut' }}
                    >
                        {isUp
                            ? <ChevronUp size={28} color={COLOR_UP} strokeWidth={3} />
                            : <ChevronDown size={28} color={COLOR_DOWN} strokeWidth={3} />}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Diagonal divider between cells */}
            {!isLast && (
                <div
                    className="absolute pointer-events-none"
                    style={{
                        right: -1,
                        top: '15%',
                        bottom: '15%',
                        width: 1,
                        background: COLOR_DIVIDER,
                        transform: 'skewX(-10deg)',
                    }}
                />
            )}
        </motion.div>
    );
};

// ── Info strip (TRACK / LAP / TIME / DISTANCE) ─────────────────────────

const formatTime = (s: number): string => {
    const total = Math.max(0, Math.floor(s));
    const m = Math.floor(total / 60);
    const sec = total % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
};

const InfoStrip = ({
    track,
    lap,
    totalLaps,
    timeSec,
    distance,
}: {
    track: string;
    lap: number;
    totalLaps: number;
    timeSec: number;
    distance: number;
}) => {
    const cells: Array<[string, string]> = [
        ['Track', track],
        ['Lap', `${lap}/${totalLaps}`],
        ['Time', formatTime(timeSec)],
        ['Distance', `${distance}M`],
    ];
    return (
        <div
            className="h-8 flex items-stretch"
            style={{ background: COLOR_STRIP, borderTop: `1px solid ${COLOR_DIVIDER}` }}
        >
            {cells.map(([label, value], i) => (
                <div
                    key={label}
                    className="flex-1 flex items-center gap-3 px-7"
                    style={{
                        borderRight: i < cells.length - 1 ? `1px solid ${COLOR_DIVIDER}` : undefined,
                    }}
                >
                    <span
                        style={{
                            fontFamily: FONT_DISPLAY,
                            fontSize: 9,
                            fontWeight: 600,
                            color: COLOR_MUTED,
                            letterSpacing: '0.24em',
                            textTransform: 'uppercase',
                            opacity: 0.7,
                        }}
                    >
                        {label}
                    </span>
                    <span
                        style={{
                            fontFamily: FONT_MONO,
                            fontSize: 13,
                            fontWeight: 600,
                            color: COLOR_FG,
                            letterSpacing: '-0.01em',
                            fontVariantNumeric: 'tabular-nums',
                        }}
                    >
                        {value}
                    </span>
                </div>
            ))}
        </div>
    );
};

// ── IN FOCUS floating card ─────────────────────────────────────────────

const InFocusCard = ({ horse }: { horse: Horse | null }) => {
    return (
        <AnimatePresence>
            {horse && (
                <motion.div
                    style={{
                        position: 'fixed',
                        bottom: 188,
                        right: 24,
                        width: 280,
                        background: COLOR_BG,
                        backdropFilter: 'blur(10px)',
                        WebkitBackdropFilter: 'blur(10px)',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.05), 0 12px 40px rgba(0,0,0,0.12)',
                        border: `0.5px solid ${COLOR_BORDER}`,
                        borderRadius: 6,
                        overflow: 'hidden',
                        zIndex: 30,
                    }}
                    initial={{ x: 320, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: 320, opacity: 0 }}
                    transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
                >
                    {/* Header */}
                    <div
                        className="flex items-center justify-between px-4 py-2"
                        style={{ background: 'rgba(0,0,0,0.04)', borderBottom: `1px solid ${COLOR_DIVIDER}` }}
                    >
                        <span
                            style={{
                                fontFamily: FONT_DISPLAY,
                                fontSize: 10,
                                fontWeight: 600,
                                color: COLOR_FG,
                                letterSpacing: '0.22em',
                                textTransform: 'uppercase',
                            }}
                        >
                            In Focus
                        </span>
                        <Eye size={14} color={COLOR_MUTED} />
                    </div>

                    {/* Body */}
                    <div className="flex items-center gap-4 p-4">
                        <div
                            className="flex-shrink-0 overflow-hidden"
                            style={{
                                width: 64,
                                height: 64,
                                background: 'rgba(0,0,0,0.04)',
                                boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.08)',
                                borderRadius: 4,
                            }}
                        >
                            <img
                                src={getSilkImagePath(horse.silkId)}
                                alt={`#${horse.number ?? '?'}`}
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className="flex flex-col gap-1 min-w-0 flex-1">
                            <span
                                style={{
                                    fontFamily: FONT_DISPLAY,
                                    fontSize: 20,
                                    fontWeight: 700,
                                    color: COLOR_FG,
                                    letterSpacing: '0.04em',
                                    textTransform: 'uppercase',
                                    lineHeight: 1,
                                    whiteSpace: 'nowrap',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                }}
                            >
                                {horse.jockeyName || horse.name || '—'}
                            </span>
                            <span
                                style={{
                                    fontFamily: FONT_DISPLAY,
                                    fontSize: 13,
                                    color: COLOR_MUTED,
                                    lineHeight: 1.2,
                                    whiteSpace: 'nowrap',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                }}
                            >
                                {horse.name || '—'}
                            </span>
                        </div>
                    </div>

                    {/* Footer: trainer (placeholder — no real data field yet) */}
                    <div
                        className="px-4 py-2 flex justify-between items-baseline"
                        style={{ borderTop: `1px solid ${COLOR_DIVIDER}` }}
                    >
                        <span
                            style={{
                                fontFamily: FONT_DISPLAY,
                                fontSize: 9,
                                fontWeight: 600,
                                color: COLOR_MUTED,
                                letterSpacing: '0.24em',
                                textTransform: 'uppercase',
                                opacity: 0.7,
                            }}
                        >
                            Trainer
                        </span>
                        <span
                            style={{
                                fontFamily: FONT_DISPLAY,
                                fontSize: 12,
                                fontWeight: 500,
                                color: COLOR_FG,
                            }}
                        >
                            —
                        </span>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

// ── Main ───────────────────────────────────────────────────────────────

export const RankingBoard = ({ rankings }: Props) => {
    const top4 = rankings.slice(0, 4);
    const sorted = [...top4].sort((a, b) => a.currentPosition - b.currentPosition);
    const leader = sorted[0];
    const deltas = usePositionDeltas(top4);

    const { race } = useRaceStore();
    const { activePTZCameraId } = useCameraStore();

    // Race-time ticker (1 Hz).
    const [now, setNow] = useState(Date.now());
    useEffect(() => {
        const t = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(t);
    }, []);
    const timeSec = race.startTime ? Math.max(0, (now - race.startTime) / 1000) : 0;
    const lap = leader?.currentLap ?? 1;
    const totalLaps = race.totalLaps ?? 1;
    const distance = race.trackLength ?? 2500;

    // IN FOCUS — pick jockey on the active PTZ camera, fallback to leader.
    const inFocus: Horse | null = activePTZCameraId
        ? top4.find(h => h.lastCameraId === activePTZCameraId) ?? leader ?? null
        : null;

    return (
        <>
            <InFocusCard horse={inFocus} />
            {/* Floating centered card — 65vw / max 1200px, bottom-anchored
                to viewport. Self-positions via position:fixed so any wrapper
                in PublicDisplay (absolute bottom-0) is ignored. */}
            <div
                style={{
                    position: 'fixed',
                    bottom: 24,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '65vw',
                    maxWidth: 1200,
                    minWidth: 720,
                    borderRadius: 16,
                    overflow: 'hidden',
                    boxShadow:
                        '0 1px 3px rgba(0,0,0,0.05), 0 12px 40px rgba(0,0,0,0.12)',
                    border: `0.5px solid ${COLOR_BORDER}`,
                    zIndex: 20,
                }}
            >
                {/* Main ranking bar */}
                <div
                    className="h-[120px] flex items-stretch"
                    style={{
                        background: COLOR_BG,
                        backdropFilter: 'blur(10px)',
                        WebkitBackdropFilter: 'blur(10px)',
                    }}
                >
                    <LayoutGroup id="ranking-board">
                        {sorted.map((h, i) => (
                            <JockeyCell
                                key={h.id}
                                horse={h}
                                isLeader={i === 0}
                                delta={deltas[h.id] || 0}
                                isLast={i === sorted.length - 1}
                            />
                        ))}
                    </LayoutGroup>
                </div>
                {/* Info strip — bottom section of the same floating block */}
                <InfoStrip
                    track="Good to Firm"
                    lap={lap}
                    totalLaps={totalLaps}
                    timeSec={timeSec}
                    distance={distance}
                />
            </div>
        </>
    );
};
