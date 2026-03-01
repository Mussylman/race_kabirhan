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

// 25 analytics cameras covering 2500m track
// Each camera covers a segment defined by track_start/track_end (matches cameras_example.json)
// go2rtcId matches stream names in go2rtc.yaml for WebRTC display
export const ANALYTICS_CAMERAS: AnalyticsCameraConfig[] = [
    { id: 'cam-01', name: 'Camera 1',  go2rtcId: 'cam-01', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.32:554/Streaming/Channels/102', trackStart: 0,    trackEnd: 110,  status: 'offline' },
    { id: 'cam-02', name: 'Camera 2',  go2rtcId: 'cam-02', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.16:554/Streaming/Channels/102', trackStart: 100,  trackEnd: 210,  status: 'offline' },
    { id: 'cam-03', name: 'Camera 3',  go2rtcId: 'cam-03', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.46:554/Streaming/Channels/102', trackStart: 200,  trackEnd: 310,  status: 'offline' },
    { id: 'cam-04', name: 'Camera 4',  go2rtcId: 'cam-04', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.44:554/Streaming/Channels/102', trackStart: 300,  trackEnd: 410,  status: 'offline' },
    { id: 'cam-05', name: 'Camera 5',  go2rtcId: 'cam-05', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.17:554/Streaming/Channels/102', trackStart: 400,  trackEnd: 510,  status: 'offline' },
    { id: 'cam-06', name: 'Camera 6',  go2rtcId: 'cam-06', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.18:554/Streaming/Channels/102', trackStart: 500,  trackEnd: 610,  status: 'offline' },
    { id: 'cam-07', name: 'Camera 7',  go2rtcId: 'cam-07', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.25:554/Streaming/Channels/102', trackStart: 600,  trackEnd: 710,  status: 'offline' },
    { id: 'cam-08', name: 'Camera 8',  go2rtcId: 'cam-08', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.34:554/Streaming/Channels/102', trackStart: 700,  trackEnd: 810,  status: 'offline' },
    { id: 'cam-09', name: 'Camera 9',  go2rtcId: 'cam-09', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.26:554/Streaming/Channels/102', trackStart: 800,  trackEnd: 910,  status: 'offline' },
    { id: 'cam-10', name: 'Camera 10', go2rtcId: 'cam-10', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.33:554/Streaming/Channels/102', trackStart: 900,  trackEnd: 1010, status: 'offline' },
    { id: 'cam-11', name: 'Camera 11', go2rtcId: 'cam-11', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.31:554/Streaming/Channels/102', trackStart: 1000, trackEnd: 1110, status: 'offline' },
    { id: 'cam-12', name: 'Camera 12', go2rtcId: 'cam-12', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.30:554/Streaming/Channels/102', trackStart: 1100, trackEnd: 1210, status: 'offline' },
    { id: 'cam-13', name: 'Camera 13', go2rtcId: 'cam-13', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.23:554/Streaming/Channels/102', trackStart: 1200, trackEnd: 1310, status: 'offline' },
    { id: 'cam-14', name: 'Camera 14', go2rtcId: 'cam-14', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.42:554/Streaming/Channels/102', trackStart: 1300, trackEnd: 1410, status: 'offline' },
    { id: 'cam-15', name: 'Camera 15', go2rtcId: 'cam-15', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.27:554/Streaming/Channels/102', trackStart: 1400, trackEnd: 1510, status: 'offline' },
    { id: 'cam-16', name: 'Camera 16', go2rtcId: 'cam-16', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.21:554/Streaming/Channels/102', trackStart: 1500, trackEnd: 1610, status: 'offline' },
    { id: 'cam-17', name: 'Camera 17', go2rtcId: 'cam-17', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.20:554/Streaming/Channels/102', trackStart: 1600, trackEnd: 1710, status: 'offline' },
    { id: 'cam-18', name: 'Camera 18', go2rtcId: 'cam-18', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.19:554/Streaming/Channels/102', trackStart: 1700, trackEnd: 1810, status: 'offline' },
    { id: 'cam-19', name: 'Camera 19', go2rtcId: 'cam-19', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.28:554/Streaming/Channels/102', trackStart: 1800, trackEnd: 1910, status: 'offline' },
    { id: 'cam-20', name: 'Camera 20', go2rtcId: 'cam-20', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.15:554/Streaming/Channels/102', trackStart: 1900, trackEnd: 2010, status: 'offline' },
    { id: 'cam-21', name: 'Camera 21', go2rtcId: 'cam-21', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.22:554/Streaming/Channels/102', trackStart: 2000, trackEnd: 2110, status: 'offline' },
    { id: 'cam-22', name: 'Camera 22', go2rtcId: 'cam-22', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.48:554/Streaming/Channels/102', trackStart: 2100, trackEnd: 2210, status: 'offline' },
    { id: 'cam-23', name: 'Camera 23', go2rtcId: 'cam-23', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.24:554/Streaming/Channels/102', trackStart: 2200, trackEnd: 2310, status: 'offline' },
    { id: 'cam-24', name: 'Camera 24', go2rtcId: 'cam-24', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.29:554/Streaming/Channels/102', trackStart: 2300, trackEnd: 2410, status: 'offline' },
    { id: 'cam-25', name: 'Camera 25', go2rtcId: 'cam-25', rtspUrl: 'rtsp://admin:Zxcv2Zxcv2@10.223.70.45:554/Streaming/Channels/102', trackStart: 2400, trackEnd: 2500, status: 'offline' },
];

// ============================================================
// PTZ CAMERAS — shown to viewers on public display
// These provide the broadcast-quality live feed.
// Operator can switch between them.
// Video via go2rtc WebRTC (passthrough H.264, no transcoding)
// ============================================================

export const PTZ_CAMERAS: PTZCameraConfig[] = [
    {
        id: 'ptz-1',
        name: 'PTZ 1 — Start/Finish',
        go2rtcId: 'ptz-1',
        rtspUrl: 'rtsp://admin:Turkestan707@10.223.70.43:554/Streaming/Channels/101',
        position: 0,
        status: 'offline',
    },
    // {
    //     id: 'ptz-2',
    //     name: 'PTZ 2 — First Turn',
    //     go2rtcId: 'ptz-2',
    //     rtspUrl: 'rtsp://admin:Turkestan707@10.223.70.44:554/Streaming/Channels/102',
    //     position: 625,
    //     status: 'offline',
    // },
    // {
    //     id: 'ptz-3',
    //     name: 'PTZ 3 — Back Straight',
    //     go2rtcId: 'ptz-3',
    //     rtspUrl: 'rtsp://admin:Turkestan707@10.223.70.45:554/Streaming/Channels/102',
    //     position: 1250,
    //     status: 'offline',
    // },
    // {
    //     id: 'ptz-4',
    //     name: 'PTZ 4 — Final Turn',
    //     go2rtcId: 'ptz-4',
    //     rtspUrl: 'rtsp://admin:Turkestan707@10.223.70.46:554/Streaming/Channels/102',
    //     position: 1875,
    //     status: 'offline',
    // },
];
