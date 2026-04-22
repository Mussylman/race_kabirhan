import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
    analyticsApi,
    type CameraInfo,
    type TimelineData,
} from '../../services/analyticsApi';

/**
 * TimelineView — horizontal timeline where each row is a camera and each
 * cell is a time bucket (default 1s). Cells colored by dominant detection
 * color that second. Click → open frame in popup.
 *
 * UX goals:
 *   - See the whole race in one view: which cameras got detections when
 *   - Spot gaps and misclassifications at a glance
 *   - Click a cell to jump into frame detail
 */

const COLOR_MAP: Record<string, string> = {
    red: 'bg-red-500',
    yellow: 'bg-yellow-400',
    green: 'bg-green-500',
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    unknown: 'bg-gray-500',
};

const BG_MAP: Record<string, string> = {
    red: '#ef4444',
    yellow: '#facc15',
    green: '#22c55e',
    blue: '#3b82f6',
    purple: '#a855f7',
    unknown: '#6b7280',
};

interface PopupState {
    cam: string;
    t: number;
    detFrames: { seq: number; colors: Record<string, number> }[];
}

export const TimelineView = () => {
    const { recId } = useParams<{ recId: string }>();
    const [cameras, setCameras] = useState<CameraInfo[]>([]);
    const [timeline, setTimeline] = useState<TimelineData>({});
    const [bucketS, setBucketS] = useState(1.0);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [popup, setPopup] = useState<PopupState | null>(null);

    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!recId) return;
        setLoading(true);
        setError(null);
        Promise.all([
            analyticsApi.listCameras(recId),
            analyticsApi.timeline(recId, bucketS),
        ])
            .then(([cams, tl]) => {
                setCameras(cams);
                setTimeline(tl);
            })
            .catch(err => setError(String(err)))
            .finally(() => setLoading(false));
    }, [recId, bucketS]);

    const maxT = useMemo(() => {
        let m = 0;
        for (const buckets of Object.values(timeline)) {
            for (const b of buckets) if (b.t > m) m = b.t;
        }
        return Math.ceil(m / 10) * 10 + 10;
    }, [timeline]);

    const pxPerSec = 6; // visual density; timeline width = maxT * pxPerSec
    const totalWidth = Math.max(800, maxT * pxPerSec);

    const onCellClick = async (cam: string, t: number) => {
        if (!recId) return;
        try {
            const dets = await analyticsApi.detections(recId, cam, t, t + bucketS);
            // Group by frame
            const byFrame: Record<number, Record<string, number>> = {};
            for (const d of dets) {
                if (!byFrame[d.frame_seq]) byFrame[d.frame_seq] = {};
                const c = d.color ?? 'unknown';
                byFrame[d.frame_seq][c] = (byFrame[d.frame_seq][c] ?? 0) + 1;
            }
            const frames = Object.entries(byFrame)
                .map(([seq, colors]) => ({ seq: Number(seq), colors }))
                .sort((a, b) => a.seq - b.seq);
            setPopup({ cam, t, detFrames: frames });
        } catch (e) {
            console.error(e);
        }
    };

    if (loading) return <div className="p-8 text-white/60">Loading timeline…</div>;
    if (error) return <div className="p-8 text-red-400">Error: {error}</div>;
    if (!cameras.length) return <div className="p-8 text-white/60">No cameras indexed yet.</div>;

    return (
        <div className="h-full flex flex-col">
            <header className="p-4 border-b border-white/10 flex items-center gap-4">
                <h2 className="text-lg font-semibold">Timeline · {recId}</h2>
                <label className="text-sm text-white/60 ml-auto">
                    Bucket:
                    <select
                        value={bucketS}
                        onChange={e => setBucketS(Number(e.target.value))}
                        className="ml-2 bg-black/40 border border-white/10 rounded px-2 py-0.5"
                    >
                        <option value={0.5}>0.5 s</option>
                        <option value={1}>1 s</option>
                        <option value={2}>2 s</option>
                        <option value={5}>5 s</option>
                    </select>
                </label>
                <Legend />
            </header>

            <div ref={scrollRef} className="flex-1 overflow-auto">
                <div style={{ width: totalWidth + 120, minWidth: '100%' }}>
                    <TimeAxis maxT={maxT} pxPerSec={pxPerSec} />
                    {cameras.map(cam => (
                        <CameraRow
                            key={cam.cam_id}
                            cam={cam}
                            buckets={timeline[cam.cam_id] ?? []}
                            bucketS={bucketS}
                            pxPerSec={pxPerSec}
                            totalWidth={totalWidth}
                            onCellClick={onCellClick}
                        />
                    ))}
                </div>
            </div>

            {popup && (
                <DetailPopup
                    recId={recId!}
                    popup={popup}
                    onClose={() => setPopup(null)}
                />
            )}
        </div>
    );
};

const Legend = () => (
    <div className="flex items-center gap-3 text-xs text-white/60">
        {Object.entries(COLOR_MAP).map(([name, klass]) => (
            <span key={name} className="flex items-center gap-1">
                <span className={`w-3 h-3 rounded-sm ${klass}`} />
                {name}
            </span>
        ))}
    </div>
);

const TimeAxis = ({ maxT, pxPerSec }: { maxT: number; pxPerSec: number }) => {
    const ticks = [];
    const step = maxT > 300 ? 30 : maxT > 120 ? 10 : 5;
    for (let t = 0; t <= maxT; t += step) {
        ticks.push(t);
    }
    return (
        <div className="flex items-end h-8 border-b border-white/10 sticky top-0 bg-[#0a0a0a] z-10">
            <div className="w-[120px] px-2 text-xs text-white/40 flex-shrink-0">time →</div>
            <div className="relative" style={{ width: maxT * pxPerSec }}>
                {ticks.map(t => (
                    <div
                        key={t}
                        className="absolute text-[10px] text-white/40"
                        style={{ left: t * pxPerSec, bottom: 2 }}
                    >
                        {t}s
                    </div>
                ))}
            </div>
        </div>
    );
};

interface CameraRowProps {
    cam: CameraInfo;
    buckets: { t: number; colors: Record<string, number>; dets: number }[];
    bucketS: number;
    pxPerSec: number;
    totalWidth: number;
    onCellClick: (cam: string, t: number) => void;
}

const CameraRow = ({ cam, buckets, bucketS, pxPerSec, onCellClick }: CameraRowProps) => {
    const cellW = Math.max(2, bucketS * pxPerSec);
    return (
        <div className="flex items-center h-8 border-b border-white/5 hover:bg-white/5">
            <div className="w-[120px] flex-shrink-0 px-2 text-sm flex items-center justify-between">
                <span className="font-mono">{cam.cam_id}</span>
                <span className="text-[10px] text-white/40">{cam.n_detections}</span>
            </div>
            <div className="relative h-6 flex-1">
                {buckets.map(b => {
                    const dominant = Object.entries(b.colors)
                        .sort(([, a], [, c]) => c - a)[0]?.[0] ?? 'unknown';
                    const bg = BG_MAP[dominant] ?? BG_MAP.unknown;
                    const opacity = Math.min(1, b.dets / 10);
                    return (
                        <div
                            key={b.t}
                            className="absolute h-5 cursor-pointer rounded-sm hover:ring-1 hover:ring-white"
                            style={{
                                left: b.t * pxPerSec,
                                width: cellW,
                                background: bg,
                                opacity: 0.4 + 0.6 * opacity,
                                top: 2,
                            }}
                            title={`${cam.cam_id} t=${b.t.toFixed(1)}s · ${b.dets} dets`}
                            onClick={() => onCellClick(cam.cam_id, b.t)}
                        />
                    );
                })}
            </div>
        </div>
    );
};

const DetailPopup = ({
    recId,
    popup,
    onClose,
}: {
    recId: string;
    popup: PopupState;
    onClose: () => void;
}) => {
    const firstSeq = popup.detFrames[0]?.seq;
    return (
        <div
            className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div
                className="bg-[#111] border border-white/10 rounded-lg max-w-4xl max-h-[85vh] overflow-auto p-6"
                onClick={e => e.stopPropagation()}
            >
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">
                        {popup.cam} · t={popup.t.toFixed(1)}s
                    </h3>
                    <button
                        onClick={onClose}
                        className="text-white/60 hover:text-white"
                    >
                        ✕
                    </button>
                </div>

                {firstSeq !== undefined && (
                    <img
                        src={analyticsApi.frameUrl(recId, popup.cam, firstSeq, true)}
                        alt=""
                        className="max-w-full max-h-[60vh] rounded border border-white/10"
                    />
                )}

                <div className="mt-4 text-sm text-white/70">
                    {popup.detFrames.length} frames in this bucket.
                    First frame: <span className="font-mono">{firstSeq}</span>
                </div>
            </div>
        </div>
    );
};
