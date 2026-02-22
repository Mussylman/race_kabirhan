// Camera Configuration for Race Vision
// Two separate configs: Analytics (backend detection) and PTZ (public display)

// ============================================================
// TYPES
// ============================================================

export type AnalyticsCameraConfig = {
    id: string;
    name: string;
    rtspUrl: string;
    trackStart: number;   // meters — start of track segment this camera covers
    trackEnd: number;     // meters — end of track segment
    status: 'online' | 'offline';
};

export type PTZCameraConfig = {
    id: string;
    name: string;
    rtspUrl: string;
    mjpegUrl: string;     // MJPEG stream URL for public display
    position: number;     // meters — position on track
    status: 'online' | 'offline';
};

// ============================================================
// BACKEND URL
// ============================================================

const BACKEND_URL = 'http://localhost:8000';

// ============================================================
// ANALYTICS CAMERAS — processed by backend (YOLO + CNN detection)
// Viewers NEVER see these streams. Only the detection results
// (jockey positions, rankings) are sent to the frontend.
// ============================================================

// Generate 25 analytics cameras evenly distributed around 2500m track
// Each camera covers ~110m (100m segment + 10m overlap)
function generateAnalyticsCameras(): AnalyticsCameraConfig[] {
    const TRACK_LENGTH = 2500;
    const NUM_CAMERAS = 25;
    const SEGMENT = TRACK_LENGTH / NUM_CAMERAS; // 100m per camera
    const OVERLAP = 10; // 10m overlap with adjacent camera

    return Array.from({ length: NUM_CAMERAS }, (_, i) => {
        const num = i + 1;
        const trackStart = i * SEGMENT;
        const trackEnd = Math.min(trackStart + SEGMENT + OVERLAP, TRACK_LENGTH);
        return {
            id: `cam-${String(num).padStart(2, '0')}`,
            name: `Analytics Camera ${num}`,
            rtspUrl: `rtsp://admin:password@192.168.1.${100 + num}:554/stream`,
            trackStart,
            trackEnd,
            status: 'offline' as const,
        };
    });
}

export const ANALYTICS_CAMERAS: AnalyticsCameraConfig[] = generateAnalyticsCameras();

// ============================================================
// PTZ CAMERAS — shown to viewers on public display
// These provide the broadcast-quality live feed.
// Operator can switch between them.
// ============================================================

export const PTZ_CAMERAS: PTZCameraConfig[] = [
    {
        id: 'ptz-1',
        name: 'PTZ Camera 1 — Start/Finish',
        rtspUrl: 'rtsp://admin:password@192.168.1.201:554/stream',
        mjpegUrl: `${BACKEND_URL}/stream/cam1`,
        position: 0,
        status: 'offline',
    },
    {
        id: 'ptz-2',
        name: 'PTZ Camera 2 — First Turn',
        rtspUrl: 'rtsp://admin:password@192.168.1.202:554/stream',
        mjpegUrl: `${BACKEND_URL}/stream/cam2`,
        position: 625,
        status: 'offline',
    },
    {
        id: 'ptz-3',
        name: 'PTZ Camera 3 — Back Straight',
        rtspUrl: 'rtsp://admin:password@192.168.1.203:554/stream',
        mjpegUrl: `${BACKEND_URL}/stream/cam3`,
        position: 1250,
        status: 'offline',
    },
    {
        id: 'ptz-4',
        name: 'PTZ Camera 4 — Final Turn',
        rtspUrl: 'rtsp://admin:password@192.168.1.204:554/stream',
        mjpegUrl: `${BACKEND_URL}/stream/cam4`,
        position: 1875,
        status: 'offline',
    },
];
