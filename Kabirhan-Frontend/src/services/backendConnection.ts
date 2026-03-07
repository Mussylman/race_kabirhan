// Backend Connection Service
// This service handles WebSocket connection to the AI backend

import { useRaceStore } from '../store/raceStore';
import { useCameraStore } from '../store/cameraStore';
import { findClosestSilkId, getSilkColor } from '../utils/silkUtils';

// Derive WebSocket URL from current page origin (works behind nginx proxy)
const getWsUrl = (): string => {
    const envUrl = import.meta.env.VITE_WS_URL;
    if (envUrl && envUrl !== '__AUTO__') return envUrl;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws`;
};

// Configuration
const CONFIG = {
    WS_URL: getWsUrl(),
    RECONNECT_DELAY: 3000,              // Retry connection every 3 seconds
    HEARTBEAT_INTERVAL: 5000,           // Send ping every 5 seconds
};

// Connection state
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
let isConnected = false;

// Connection status callback
let onStatusChange: ((status: 'connecting' | 'connected' | 'disconnected' | 'error') => void) | null = null;

// Set status change callback
export const setConnectionStatusCallback = (callback: typeof onStatusChange) => {
    onStatusChange = callback;
};

// Connect to backend
export const connectToBackend = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('Already connected to backend');
        return;
    }

    console.log(`🔌 Connecting to backend: ${CONFIG.WS_URL}`);
    onStatusChange?.('connecting');

    try {
        ws = new WebSocket(CONFIG.WS_URL);

        ws.onopen = () => {
            console.log('✅ Connected to AI backend');
            isConnected = true;
            onStatusChange?.('connected');

            // Start heartbeat
            heartbeatTimer = setInterval(() => {
                if (ws?.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, CONFIG.HEARTBEAT_INTERVAL);

            // Request initial state
            if (ws) {
                ws.send(JSON.stringify({ type: 'get_state' }));
            }
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                handleBackendMessage(message);
            } catch (error) {
                console.error('Failed to parse message:', error);
            }
        };

        ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            onStatusChange?.('error');
        };

        ws.onclose = () => {
            console.log('🔌 Disconnected from backend');
            isConnected = false;
            onStatusChange?.('disconnected');

            // Clear heartbeat
            if (heartbeatTimer) clearInterval(heartbeatTimer);

            // Schedule reconnect
            reconnectTimer = setTimeout(() => {
                console.log('🔄 Attempting to reconnect...');
                connectToBackend();
            }, CONFIG.RECONNECT_DELAY);
        };

    } catch (error) {
        console.error('Failed to connect:', error);
        onStatusChange?.('error');
    }
};

// Disconnect from backend
export const disconnectFromBackend = () => {
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
    isConnected = false;
    console.log('🔌 Disconnected from backend');
};

// Send message to backend
export const sendToBackend = (message: object) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    } else {
        console.warn('Cannot send message: not connected to backend');
    }
};

// Handle incoming messages from backend
const handleBackendMessage = (message: BackendMessage) => {
    const { updateRankings, startRace, stopRace, setRaceConfig, addHorse, clearHorses } = useRaceStore.getState();
    const { updateAnalyticsCameraHorses, setActivePTZ } = useCameraStore.getState();

    switch (message.type) {
        // ============ RACE EVENTS ============
        case 'race_start':
            console.log('🏁 Race started from backend');
            if (message.race) {
                setRaceConfig({
                    name: message.race.name,
                    totalLaps: message.race.totalLaps,
                    trackLength: message.race.trackLength,
                });
            }
            startRace();
            break;

        case 'race_stop':
        case 'race_finish':
            console.log('🏁 Race finished');
            stopRace();
            break;

        // ============ HORSE EVENTS ============
        case 'horses_detected':
            // Backend detected horses at race start
            console.log(`🏇 ${message.horses?.length} horses detected`);
            clearHorses();
            message.horses?.forEach((horse: HorseData) => {
                const detectedColor = horse.color || getDefaultColor(horse.number);
                const silkId = horse.silkId || findClosestSilkId(detectedColor);
                addHorse({
                    id: horse.id,
                    name: horse.name || `Horse ${horse.number}`,
                    number: horse.number,
                    color: detectedColor,
                    jockeyName: horse.jockeyName || `Jockey ${horse.number}`,
                    silkId: silkId,
                    silkColor: getSilkColor(silkId),
                });
            });
            break;

        case 'horse_update':
            // Single horse position update
            // This is called when a horse passes a camera
            console.log(`📷 Horse #${message.horse?.number} at camera ${message.cameraId}`);
            break;

        // ============ RANKING EVENTS ============
        case 'ranking_update':
            // Main ranking update - called when horses pass cameras
            console.log('📊 Ranking update received');
            if (message.rankings) {
                const formattedRankings = message.rankings.map((r: RankingData, index: number) => {
                    // Backend'den gelen renk ile en yakın silk'i eşleştir
                    const detectedColor = r.color || getDefaultColor(r.number);
                    const silkId = r.silkId || findClosestSilkId(detectedColor);
                    const silkColor = r.silkColor || getSilkColor(silkId);

                    return {
                        id: r.id,
                        name: r.name || `Horse ${r.number}`,
                        number: r.number,
                        color: detectedColor,
                        jockeyName: r.jockeyName || `Jockey ${r.number}`,
                        currentPosition: r.position || index + 1,
                        distanceCovered: r.distanceCovered || 0,
                        currentLap: r.currentLap || 1,
                        timeElapsed: r.timeElapsed || 0,
                        speed: r.speed || 0,
                        gapToLeader: r.gapToLeader || 0,
                        lastCameraId: r.lastCameraId || '',
                        silkId: silkId,
                        silkColor: silkColor,
                    };
                });
                updateRankings(formattedRankings);
            }
            break;

        // ============ CAMERA EVENTS ============
        case 'camera_detection':
            // Horses detected in a specific camera
            if (message.cameraId && message.horseIds) {
                updateAnalyticsCameraHorses(message.cameraId, message.horseIds);
            }
            break;

        case 'camera_switch':
            // PTZ camera switched
            if (message.cameraId) {
                setActivePTZ(message.cameraId);
            }
            break;

        case 'activation_map':
            // Camera activation map from multi-camera pipeline
            // {cameras: {cam_id: active_bool}}
            if (message.cameras) {
                console.log('📷 Activation map:', Object.values(message.cameras).filter(Boolean).length, 'active');
                const { updateCameraActivation } = useCameraStore.getState();
                if (typeof updateCameraActivation === 'function') {
                    updateCameraActivation(message.cameras as Record<string, boolean>);
                }
            }
            break;

        case 'live_detections':
            // Real-time detection status from DeepStream C++ (with colors)
            if (message.cameras) {
                const { setLiveDetections } = useCameraStore.getState();
                if (typeof setLiveDetections === 'function') {
                    setLiveDetections(message.cameras as unknown as Record<string, Array<{color: string, conf: number, track_id?: number}>>);
                }
            }
            break;

        case 'camera_result':
            // Single camera finished analysis — ranking from vote engine
            console.log(`📷 Camera ${message.cameraId} result:`, message.ranking, `(${message.voteFrames} frames)`);
            break;

        case 'camera_status':
            // Full camera status from backend
            console.log('📷 Camera status:', message.total_analytics, 'analytics,', message.active_analytics, 'active');
            break;

        // ============ SYSTEM EVENTS ============
        case 'pong':
            // Heartbeat response
            break;

        case 'state':
            // Full state sync
            console.log('📦 Received full state from backend');
            if (message.race) {
                setRaceConfig(message.race);
            }
            if (message.rankings) {
                handleBackendMessage({ type: 'ranking_update', rankings: message.rankings });
            }
            break;

        case 'error':
            console.error('❌ Backend error:', message.message);
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
};

// Helper function for default horse colors
const getDefaultColor = (number: number): string => {
    const colors = [
        '#EF4444', '#F97316', '#EAB308', '#22C55E', '#14B8A6',
        '#3B82F6', '#8B5CF6', '#EC4899', '#6366F1', '#06B6D4'
    ];
    return colors[(number - 1) % colors.length];
};

// Get connection status
export const isBackendConnected = () => isConnected;

// ============ MESSAGE TYPES ============

interface BackendMessage {
    type: string;
    race?: {
        name: string;
        totalLaps: number;
        trackLength: number;
        status?: string;
    };
    horses?: HorseData[];
    horse?: HorseData;
    rankings?: RankingData[];
    cameraId?: string;
    horseIds?: string[];
    message?: string;
    // Multi-camera fields
    cameras?: Record<string, boolean> | CameraStatusEntry[];
    total_analytics?: number;
    active_analytics?: number;
    total_display?: number;
    // Camera result fields
    ranking?: string[];
    voteFrames?: number;
}

interface CameraStatusEntry {
    cam_id: string;
    role: string;
    active: boolean;
    connected: boolean;
    track_segment: string | null;
}

interface HorseData {
    id: string;
    number: number;
    name?: string;
    color?: string;
    jockeyName?: string;
    silkId?: number;
    silkColor?: string;
}

interface RankingData {
    id: string;
    number: number;
    name?: string;
    color?: string;
    jockeyName?: string;
    position?: number;
    distanceCovered?: number;
    currentLap?: number;
    timeElapsed?: number;
    speed?: number;
    gapToLeader?: number;
    lastCameraId?: string;
    silkId?: number;
    silkColor?: string;
}
