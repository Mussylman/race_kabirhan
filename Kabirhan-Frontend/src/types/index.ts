// Horse (At) interface
export interface Horse {
    id: string;
    name: string;
    number: number;
    color: string;
    jockeyName: string;
    currentPosition: number;
    currentLap: number;
    timeElapsed: number;
    gapToLeader: number;
    distanceCovered: number;
    speed: number;
    lastCameraId: string;
    // Silk (Jokey Kıyafeti) alanları
    silkId: number;      // 1-10 arası silk ID
    silkColor: string;   // Silk'in dominant rengi
}

// Camera interface
export interface Camera {
    id: string;
    type: 'fixed' | 'ptz';
    position: number;
    status: 'online' | 'offline';
    streamUrl: string;
}

// PTZ Camera (extends Camera)
export interface PTZCamera extends Camera {
    type: 'ptz';
    isActive: boolean;
}

// Fixed Camera (extends Camera)
export interface FixedCamera extends Camera {
    type: 'fixed';
    horsesInView: string[];
}

// Race Configuration
export interface RaceConfig {
    id: string;
    name: string;
    totalLaps: number;
    trackLength: number;
    horses: Horse[];
    status: 'pending' | 'active' | 'finished';
    startTime: number | null;
    currentTime: number;
    startFinishPosition: number; // 0-2500m - where start/finish line is on track
}

// WebSocket Message Types
export interface RankingUpdate {
    type: 'ranking_update';
    data: {
        rankings: Array<{
            horseId: string;
            position: number;
            lap: number;
            timeElapsed: number;
            gapToLeader: number;
        }>;
    };
}

export interface HorsePositionUpdate {
    type: 'horse_position';
    data: {
        horseId: string;
        position: number;
        lap: number;
        cameraId: string;
        speed: number;
        distanceCovered: number;
    };
}

export interface CameraStatusUpdate {
    type: 'camera_status';
    data: {
        cameraId: string;
        status: 'online' | 'offline';
    };
}

// Constants
export const TRACK_LENGTH = 2500;
export const PTZ_CAMERA_COUNT = 4;
export const MAX_HORSES = 5;
export const TOP_HORSES_DISPLAY = 10;
export const TOTAL_SILKS = 10;

// Silk (Jockey Outfit) colors - Dominant color for each silk ID
export const SILK_COLORS: Record<number, string> = {
    1: '#DC2626',  // Red
    2: '#2563EB',  // Blue
    3: '#16A34A',  // Green
    4: '#FBBF24',  // Yellow
    5: '#9333EA',  // Purple
    6: '#EA580C',  // Orange
    7: '#EC4899',  // Pink
    8: '#06B6D4',  // Cyan
    9: '#84CC16',  // Lime
    10: '#F97316', // Orange variant
};

// Color palette for horses
export const HORSE_COLORS = [
    '#DC2626', '#2563EB', '#16A34A', '#FBBF24', '#9333EA',
    '#EA580C', '#EC4899', '#06B6D4', '#84CC16', '#F97316',
    '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#3B82F6',
    '#14B8A6', '#F472B6', '#A855F7', '#22C55E', '#FB923C'
];

// Sample horse names
export const HORSE_NAMES = [
    'Thunder Bolt', 'Lightning Strike', 'Storm Chaser', 'Wind Runner',
    'Fire Spirit', 'Golden Arrow', 'Silver Bullet', 'Dark Knight',
    'Royal Crown', 'Swift Shadow', 'Blazing Star', 'Midnight Express'
];

// Sample jockey names
export const JOCKEY_NAMES = [
    'John Smith', 'Mike Johnson', 'Alex Turner', 'Chris Davis',
    'Ryan Wilson', 'David Lee', 'James Brown', 'Tom Carter',
    'Mark Robinson', 'Steve Miller', 'Kevin White', 'Paul Harris'
];
