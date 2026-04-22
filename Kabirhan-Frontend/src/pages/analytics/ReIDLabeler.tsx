import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { useParams } from 'react-router-dom';
import {
    analyticsApi,
    type LabelStrategy,
    type Pair,
    type PairStats,
} from '../../services/analyticsApi';

/**
 * ReIDLabeler — side-by-side cross-camera pair labeling with active-learning
 * priority queue.
 *
 * Hotkeys:
 *   1 / S — same jockey
 *   2 / D — different
 *   3 / K — skip
 *   Z     — undo last
 *   R     — reload current pair (if anything is stuck)
 *
 * Queue strategy defaults to "uncertainty" (hardest pairs first). Switch
 * to "auto_conf_middle" during first session when we have no model yet —
 * it picks pairs whose auto_conf is closest to 0.5.
 */

type PendingPair = Pair | { done: true };

export const ReIDLabeler = () => {
    const { recId } = useParams<{ recId: string }>();
    const [pair, setPair] = useState<Pair | null>(null);
    const [done, setDone] = useState(false);
    const [stats, setStats] = useState<PairStats | null>(null);
    const [strategy, setStrategy] = useState<LabelStrategy>('uncertainty');
    const [labeling, setLabeling] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sessionCount, setSessionCount] = useState(0);
    const lastPairIdRef = useRef<number | null>(null);

    const loadNext = useCallback(async () => {
        if (!recId) return;
        setError(null);
        try {
            const [next, st] = await Promise.all([
                analyticsApi.nextPair(recId, strategy) as Promise<PendingPair>,
                analyticsApi.pairStats(recId),
            ]);
            setStats(st);
            if ((next as { done?: true }).done) {
                setPair(null);
                setDone(true);
            } else {
                setPair(next as Pair);
                setDone(false);
            }
        } catch (err) {
            setError(String(err));
        }
    }, [recId, strategy]);

    useEffect(() => {
        loadNext();
    }, [loadNext]);

    const submit = useCallback(
        async (label: 'same' | 'different' | 'skip') => {
            if (!pair || labeling) return;
            setLabeling(true);
            lastPairIdRef.current = pair.id;
            try {
                await analyticsApi.labelPair(pair.id, label);
                setSessionCount(c => c + 1);
                await loadNext();
            } catch (err) {
                setError(String(err));
            } finally {
                setLabeling(false);
            }
        },
        [pair, labeling, loadNext]
    );

    // Keyboard shortcuts
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.repeat) return;
            const k = e.key.toLowerCase();
            if (k === '1' || k === 's') submit('same');
            else if (k === '2' || k === 'd') submit('different');
            else if (k === '3' || k === 'k') submit('skip');
            else if (k === 'r') loadNext();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [submit, loadNext]);

    if (!recId) return null;

    return (
        <div className="h-full flex flex-col">
            <header className="p-4 border-b border-white/10 flex items-center gap-4">
                <h2 className="text-lg font-semibold">ReID Labeler · {recId}</h2>
                <div className="flex items-center gap-2 text-sm text-white/60">
                    <label>Strategy:</label>
                    <select
                        value={strategy}
                        onChange={e => setStrategy(e.target.value as LabelStrategy)}
                        className="bg-black/40 border border-white/10 rounded px-2 py-0.5"
                    >
                        <option value="uncertainty">Uncertainty (active learning)</option>
                        <option value="auto_conf_middle">Auto conf ≈ 0.5</option>
                        <option value="random">Random</option>
                    </select>
                </div>
                <StatsBar stats={stats} sessionCount={sessionCount} />
            </header>

            {error && (
                <div className="p-3 bg-red-900/30 border-b border-red-800 text-red-300 text-sm">
                    {error}
                </div>
            )}

            <div className="flex-1 overflow-hidden flex items-center justify-center p-6">
                {done ? (
                    <div className="text-center">
                        <div className="text-2xl mb-2">🎉 All pairs labeled</div>
                        <div className="text-white/60">
                            Export the dataset via POST /analytics/export/market1501
                        </div>
                    </div>
                ) : pair ? (
                    <PairCanvas recId={recId} pair={pair} onLabel={submit} labeling={labeling} />
                ) : (
                    <div className="text-white/60">
                        No pairs yet. Run
                        <code className="mx-2 bg-black/40 px-1 rounded">tools/auto_label_cross_cam.py</code>
                        to populate.
                    </div>
                )}
            </div>

            <footer className="p-3 border-t border-white/10 text-xs text-white/50 flex items-center gap-4 justify-center">
                <span><kbd className="kbd">1</kbd> / <kbd className="kbd">S</kbd> Same</span>
                <span><kbd className="kbd">2</kbd> / <kbd className="kbd">D</kbd> Different</span>
                <span><kbd className="kbd">3</kbd> / <kbd className="kbd">K</kbd> Skip</span>
                <span><kbd className="kbd">R</kbd> Reload</span>
            </footer>
        </div>
    );
};

const StatsBar = ({
    stats,
    sessionCount,
}: {
    stats: PairStats | null;
    sessionCount: number;
}) => {
    if (!stats) return null;
    const labeled = (stats.same_cnt ?? 0) + (stats.diff_cnt ?? 0) + (stats.skip_cnt ?? 0);
    const pct = stats.total ? Math.round((labeled / stats.total) * 100) : 0;
    return (
        <div className="ml-auto flex items-center gap-4 text-sm">
            <span className="text-white/70">
                {labeled}/{stats.total} ({pct}%)
            </span>
            <span className="text-green-400">{stats.same_cnt} same</span>
            <span className="text-red-400">{stats.diff_cnt} diff</span>
            <span className="text-white/40">{stats.skip_cnt} skip</span>
            <span className="text-white/40">· session {sessionCount}</span>
        </div>
    );
};

interface PairCanvasProps {
    recId: string;
    pair: Pair;
    onLabel: (l: 'same' | 'different' | 'skip') => void;
    labeling: boolean;
}

const PairCanvas = ({ recId, pair, onLabel, labeling }: PairCanvasProps) => {
    const aUrl =
        pair.a_det_id != null
            ? analyticsApi.cropUrl(recId, pair.a_cam, pair.a_seq, pair.a_det_id)
            : analyticsApi.frameUrl(recId, pair.a_cam, pair.a_seq, true);
    const bUrl =
        pair.b_det_id != null
            ? analyticsApi.cropUrl(recId, pair.b_cam, pair.b_seq, pair.b_det_id)
            : analyticsApi.frameUrl(recId, pair.b_cam, pair.b_seq, true);

    const difficultyBadge =
        pair.uncertainty > 0.7
            ? { color: 'bg-amber-500', label: 'HARD' }
            : pair.uncertainty > 0.4
                ? { color: 'bg-blue-500', label: 'MEDIUM' }
                : { color: 'bg-green-600', label: 'EASY' };

    return (
        <div className="w-full max-w-5xl">
            <div className="text-center mb-4 text-sm text-white/60">
                Pair #{pair.id} ·
                <span className={`ml-2 px-2 py-0.5 rounded text-xs font-semibold ${difficultyBadge.color}`}>
                    {difficultyBadge.label}
                </span>
                <span className="ml-2">
                    auto_conf={pair.auto_conf.toFixed(2)}
                    {pair.model_sim != null && ` · model_sim=${pair.model_sim.toFixed(2)}`}
                </span>
                <span className="ml-2 text-white/40">{pair.source}</span>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <PairSide
                    imgUrl={aUrl}
                    cam={pair.a_cam}
                    seq={pair.a_seq}
                    color={pair.a_color}
                    conf={pair.a_conf}
                />
                <PairSide
                    imgUrl={bUrl}
                    cam={pair.b_cam}
                    seq={pair.b_seq}
                    color={pair.b_color}
                    conf={pair.b_conf}
                />
            </div>

            <div className="mt-6 flex gap-3 justify-center">
                <LabelButton
                    onClick={() => onLabel('same')}
                    disabled={labeling}
                    className="bg-green-600 hover:bg-green-500"
                >
                    1 · SAME
                </LabelButton>
                <LabelButton
                    onClick={() => onLabel('different')}
                    disabled={labeling}
                    className="bg-red-600 hover:bg-red-500"
                >
                    2 · DIFFERENT
                </LabelButton>
                <LabelButton
                    onClick={() => onLabel('skip')}
                    disabled={labeling}
                    className="bg-white/10 hover:bg-white/20"
                >
                    3 · SKIP
                </LabelButton>
            </div>
        </div>
    );
};

const PairSide = ({
    imgUrl,
    cam,
    seq,
    color,
    conf,
}: {
    imgUrl: string;
    cam: string;
    seq: number;
    color: string | null;
    conf: number;
}) => (
    <div className="bg-black/40 border border-white/10 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2 text-sm">
            <span className="font-mono">{cam}</span>
            <span className="text-white/40">seq {seq}</span>
        </div>
        <img
            src={imgUrl}
            alt={`${cam} seq ${seq}`}
            className="w-full rounded border border-white/10 bg-black object-contain max-h-[55vh]"
        />
        <div className="mt-2 flex items-center gap-2 text-xs">
            {color && (
                <span className="px-2 py-0.5 rounded-full bg-white/10">
                    {color} {Math.round(conf)}%
                </span>
            )}
        </div>
    </div>
);

const LabelButton = ({
    onClick,
    disabled,
    className,
    children,
}: {
    onClick: () => void;
    disabled?: boolean;
    className?: string;
    children: ReactNode;
}) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`px-6 py-3 rounded font-semibold text-sm disabled:opacity-40 ${className ?? ''}`}
    >
        {children}
    </button>
);
