import { NavLink, Outlet, useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { analyticsApi, type Recording } from '../../services/analyticsApi';

/**
 * AnalyticsLayout — wraps all /analytics/* routes with a sidebar that
 * shows the current recording, a navigation menu, and a pair-labeling
 * progress indicator.
 *
 * Routes below this layout:
 *   /analytics                        — redirect to recordings picker
 *   /analytics/:recId/timeline        — TimelineView
 *   /analytics/:recId/labeler         — ReIDLabeler
 *   /analytics/:recId/camera/:camId   — FrameBrowser (future)
 *   /analytics/:recId/journey         — JockeyJourney (future)
 *   /analytics/:recId/metrics         — MetricsDashboard (future)
 */
export const AnalyticsLayout = () => {
    const { recId } = useParams<{ recId?: string }>();
    const [recordings, setRecordings] = useState<Recording[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        analyticsApi.listRecordings()
            .then(setRecordings)
            .catch(err => console.error('listRecordings:', err))
            .finally(() => setLoading(false));
    }, []);

    const currentRec = recordings.find(r => r.rec_id === recId);

    return (
        <div className="h-screen w-screen flex bg-[#0a0a0a] text-white overflow-hidden">
            <aside className="w-64 border-r border-white/10 flex flex-col">
                <div className="p-4 border-b border-white/10">
                    <div className="text-xs uppercase text-white/40 tracking-wider">
                        Kabirhan Analytics
                    </div>
                    <div className="mt-1 font-semibold text-sm">
                        {currentRec?.rec_id ?? 'Pick a recording'}
                    </div>
                    {currentRec && (
                        <div className="mt-1 text-xs text-white/50">
                            {currentRec.n_cameras} cams · {currentRec.n_frames} frames
                        </div>
                    )}
                </div>

                <nav className="p-2 flex-1 overflow-y-auto text-sm">
                    {recId && (
                        <div className="mb-4">
                            <div className="px-2 py-1 text-[10px] uppercase text-white/40">Views</div>
                            <NavItem to={`/analytics/${recId}/timeline`} label="Timeline" />
                            <NavItem to={`/analytics/${recId}/labeler`} label="ReID Labeler" />
                            {/* Future:
                            <NavItem to={`/analytics/${recId}/camera/cam-01`} label="Frame Browser" />
                            <NavItem to={`/analytics/${recId}/journey`} label="Jockey Journey" />
                            <NavItem to={`/analytics/${recId}/metrics`} label="Metrics" />
                            */}
                        </div>
                    )}

                    <div>
                        <div className="px-2 py-1 text-[10px] uppercase text-white/40">Recordings</div>
                        {loading && <div className="px-2 text-white/40">Loading…</div>}
                        {!loading && recordings.length === 0 && (
                            <div className="px-2 text-white/40 text-xs">
                                No indexed recordings yet. Use
                                <code className="block mt-1 bg-black/40 p-1 rounded text-[10px]">
                                    POST /analytics/recordings/index
                                </code>
                            </div>
                        )}
                        {recordings.map(r => (
                            <NavItem
                                key={r.rec_id}
                                to={`/analytics/${encodeURIComponent(r.rec_id)}/timeline`}
                                label={r.rec_id}
                                small
                            />
                        ))}
                    </div>
                </nav>

                <div className="p-2 border-t border-white/10 text-xs text-white/40">
                    <a href="/operator" className="hover:text-white/80">← Back to operator</a>
                </div>
            </aside>

            <main className="flex-1 overflow-hidden">
                <Outlet />
            </main>
        </div>
    );
};

const NavItem = ({ to, label, small }: { to: string; label: string; small?: boolean }) => (
    <NavLink
        to={to}
        className={({ isActive }) =>
            `block px-2 py-1 rounded mb-0.5 ${
                isActive
                    ? 'bg-white/10 text-white'
                    : 'text-white/60 hover:bg-white/5 hover:text-white/90'
            } ${small ? 'text-xs truncate' : ''}`
        }
        title={label}
    >
        {label}
    </NavLink>
);
