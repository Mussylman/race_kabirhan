/**
 * analyticsApi.ts — typed client for the /analytics/* backend.
 *
 * All URLs are derived from window.location so the app works whether
 * served by nginx, vite dev, or inside Docker. Override via
 * VITE_ANALYTICS_URL if the analytics backend lives elsewhere.
 */

const getBaseUrl = (): string => {
    const envUrl = import.meta.env.VITE_ANALYTICS_URL;
    if (envUrl && envUrl !== '__AUTO__') return envUrl;
    // Default: same origin, analytics mounted under /analytics
    return `${window.location.protocol}//${window.location.host}`;
};

const BASE = getBaseUrl();

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE}${path}`, init);
    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return res.json() as Promise<T>;
}

// --- Types ----------------------------------------------------------

export interface Recording {
    rec_id: string;
    jsonl_dir: string;
    mp4_dir: string | null;
    n_cameras: number;
    n_frames: number;
    created_at: number;
}

export interface CameraInfo {
    cam_id: string;
    mp4_path: string | null;
    n_frames: number;
    n_detections: number;
}

export interface TimelineBucket {
    t: number;
    colors: Record<string, number>;
    dets: number;
}

export type TimelineData = Record<string, TimelineBucket[]>;

export interface Detection {
    id: number;
    rec_id: string;
    cam_id: string;
    frame_seq: number;
    ts_capture: number;
    video_sec: number;
    frame_w: number;
    frame_h: number;
    bbox_x1: number;
    bbox_y1: number;
    bbox_x2: number;
    bbox_y2: number;
    color: string | null;
    conf: number;
    track_id: number | null;
}

export interface Pair {
    id: number;
    rec_id: string;
    a_cam: string;
    a_seq: number;
    a_det_id: number | null;
    a_bbox: [number, number, number, number] | null;
    a_color: string | null;
    a_conf: number;
    b_cam: string;
    b_seq: number;
    b_det_id: number | null;
    b_bbox: [number, number, number, number] | null;
    b_color: string | null;
    b_conf: number;
    auto_conf: number;
    model_sim: number | null;
    uncertainty: number;
    human_label: 'same' | 'different' | 'skip' | null;
    source: string;
}

export interface PairStats {
    total: number;
    same_cnt: number;
    diff_cnt: number;
    skip_cnt: number;
    unlabeled: number;
}

export type LabelStrategy = 'uncertainty' | 'random' | 'auto_conf_middle';

// --- API ------------------------------------------------------------

export const analyticsApi = {
    // Recordings & index
    listRecordings: () =>
        fetchJson<Recording[]>('/analytics/recordings'),

    indexRecording: (rec_id: string, jsonl_dir: string, mp4_dir?: string) =>
        fetchJson<{ rec_id: string; cameras: number; frames: number; detections: number }>(
            '/analytics/recordings/index',
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rec_id, jsonl_dir, mp4_dir }),
            }
        ),

    listCameras: (rec_id: string) =>
        fetchJson<CameraInfo[]>(`/analytics/recordings/${encodeURIComponent(rec_id)}/cameras`),

    timeline: (rec_id: string, bucket_s = 1.0) =>
        fetchJson<TimelineData>(
            `/analytics/recordings/${encodeURIComponent(rec_id)}/timeline?bucket_s=${bucket_s}`
        ),

    detections: (rec_id: string, cam: string, t_start = 0, t_end = 9999) =>
        fetchJson<Detection[]>(
            `/analytics/recordings/${encodeURIComponent(rec_id)}/detections?cam=${cam}&t_start=${t_start}&t_end=${t_end}`
        ),

    // Frames & crops (direct image URLs, not JSON)
    frameUrl: (rec_id: string, cam: string, seq: number, annotate = true) =>
        `${BASE}/analytics/frame/${encodeURIComponent(rec_id)}/${cam}/${seq}?annotate=${annotate}`,

    cropUrl: (rec_id: string, cam: string, seq: number, det_id: number) =>
        `${BASE}/analytics/crop/${encodeURIComponent(rec_id)}/${cam}/${seq}/${det_id}`,

    // Pairs & labeler
    pairStats: (rec_id: string) =>
        fetchJson<PairStats>(`/analytics/pairs/stats?rec_id=${encodeURIComponent(rec_id)}`),

    nextPair: (rec_id: string, strategy: LabelStrategy = 'uncertainty') =>
        fetchJson<Pair | { done: true }>(
            `/analytics/pairs/next?rec_id=${encodeURIComponent(rec_id)}&strategy=${strategy}`
        ),

    labelPair: (pair_id: number, label: 'same' | 'different' | 'skip', labeler = 'operator') =>
        fetchJson<{ ok: boolean; pair_id: number }>(
            `/analytics/pairs/${pair_id}/label`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label, labeler }),
            }
        ),

    listLabeled: (rec_id: string, label?: 'same' | 'different' | 'skip') =>
        fetchJson<Pair[]>(
            `/analytics/pairs/labeled?rec_id=${encodeURIComponent(rec_id)}${label ? `&label=${label}` : ''}`
        ),

    // Export
    exportMarket1501: (rec_id: string) =>
        fetchJson<{ path: string }>(
            `/analytics/export/market1501?rec_id=${encodeURIComponent(rec_id)}`,
            { method: 'POST' }
        ),

    exportTriplets: (rec_id: string, negatives_per_anchor = 4) =>
        fetchJson<{ path: string }>(
            `/analytics/export/triplets?rec_id=${encodeURIComponent(rec_id)}&negatives_per_anchor=${negatives_per_anchor}`,
            { method: 'POST' }
        ),

    exportCoco: (rec_id: string) =>
        fetchJson<{ path: string }>(
            `/analytics/export/coco?rec_id=${encodeURIComponent(rec_id)}`,
            { method: 'POST' }
        ),
};
