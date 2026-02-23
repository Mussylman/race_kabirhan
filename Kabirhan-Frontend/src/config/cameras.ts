// Camera Configuration for Race Vision
// Two separate configs: Analytics (backend detection) and PTZ (public display)
// Video display: go2rtc (WebRTC passthrough, no transcoding)
// Analytics: DeepStream → SHM → Python → WebSocket

// ============================================================
// TYPES
// ============================================================

export type AnalyticsCameraConfig = {
    id: string;
    name: string;
    go2rtcId: string;     // go2rtc stream ID for WebRTC display
    rtspUrl: string;
    trackStart: number;   // meters — start of track segment this camera covers
    trackEnd: number;     // meters — end of track segment
    status: 'online' | 'offline';
};

export type PTZCameraConfig = {
    id: string;
    name: string;
    go2rtcId: string;     // go2rtc stream ID for WebRTC display
    rtspUrl: string;
    position: number;     // meters — position on track
    status: 'online' | 'offline';
};

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
        const padded = String(num).padStart(2, '0');
        const trackStart = i * SEGMENT;
        const trackEnd = Math.min(trackStart + SEGMENT + OVERLAP, TRACK_LENGTH);
        return {
            id: `cam-${padded}`,
            name: `Analytics Camera ${num}`,
            go2rtcId: `cam-${padded}`,
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
// Video via go2rtc WebRTC (passthrough H.264, no transcoding)
// ============================================================

export const PTZ_CAMERAS: PTZCameraConfig[] = [
    {
        id: 'ptz-1',
        name: 'PTZ Camera 1 — Start/Finish',
        go2rtcId: 'ptz-1',
        rtspUrl: 'rtsp://admin:password@192.168.1.201:554/stream',
        position: 0,
        status: 'offline',
    },
    {
        id: 'ptz-2',
        name: 'PTZ Camera 2 — First Turn',
        go2rtcId: 'ptz-2',
        rtspUrl: 'rtsp://admin:password@192.168.1.202:554/stream',
        position: 625,
        status: 'offline',
    },
    {
        id: 'ptz-3',
        name: 'PTZ Camera 3 — Back Straight',
        go2rtcId: 'ptz-3',
        rtspUrl: 'rtsp://admin:password@192.168.1.203:554/stream',
        position: 1250,
        status: 'offline',
    },
    {
        id: 'ptz-4',
        name: 'PTZ Camera 4 — Final Turn',
        go2rtcId: 'ptz-4',
        rtspUrl: 'rtsp://admin:password@192.168.1.204:554/stream',
        position: 1875,
        status: 'offline',
    },
];
