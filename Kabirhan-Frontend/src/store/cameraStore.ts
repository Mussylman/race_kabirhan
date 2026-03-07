import { create } from 'zustand';
import { ANALYTICS_CAMERAS, PTZ_CAMERAS } from '../config/cameras';
import type { AnalyticsCameraConfig, PTZCameraConfig } from '../config/cameras';

// Store camera types (extend config with runtime state)
export interface AnalyticsCameraState extends AnalyticsCameraConfig {
    horsesInView: string[];
}

export interface PTZCameraState extends PTZCameraConfig {
    isActive: boolean;
}

interface CameraState {
    ptzCameras: PTZCameraState[];
    analyticsCameras: AnalyticsCameraState[];
    activePTZCameraId: string;
    liveDetections: Record<string, Array<{color: string, conf: number, track_id?: number}>>;  // cam_id → detections with colors

    // Actions
    setActivePTZCamera: (cameraId: string) => void;
    setActivePTZ: (cameraId: string) => void;
    updateCameraStatus: (cameraId: string, status: 'online' | 'offline') => void;
    updateAnalyticsCameraHorses: (cameraId: string, horseIds: string[]) => void;
    updateCameraActivation: (cameras: Record<string, boolean>) => void;
    setLiveDetections: (cameras: Record<string, Array<{color: string, conf: number, track_id?: number}>>) => void;
    initializeCameras: () => void;
    syncFromStorage: () => void;
}

// Build PTZ cameras from config
const createPTZCameras = (): PTZCameraState[] => {
    const savedCamera = getSavedCamera();
    return PTZ_CAMERAS.map(cam => ({
        ...cam,
        isActive: cam.id === savedCamera,
    }));
};

// Build analytics cameras from config
const createAnalyticsCameras = (): AnalyticsCameraState[] => {
    return ANALYTICS_CAMERAS.map(cam => ({
        ...cam,
        horsesInView: [],
    }));
};

// Get saved camera from localStorage
const getSavedCamera = (): string => {
    return localStorage.getItem('activePTZCamera') || PTZ_CAMERAS[0]?.id || 'ptz-1';
};

export const useCameraStore = create<CameraState>((set, get) => ({
    ptzCameras: createPTZCameras(),
    analyticsCameras: createAnalyticsCameras(),
    activePTZCameraId: getSavedCamera(),
    liveDetections: {},

    setActivePTZCamera: (cameraId) => {
        localStorage.setItem('activePTZCamera', cameraId);
        set((state) => ({
            ptzCameras: state.ptzCameras.map(cam => ({
                ...cam,
                isActive: cam.id === cameraId
            })),
            activePTZCameraId: cameraId
        }));
    },

    setActivePTZ: (cameraId) => {
        localStorage.setItem('activePTZCamera', cameraId);
        set((state) => ({
            ptzCameras: state.ptzCameras.map(cam => ({
                ...cam,
                isActive: cam.id === cameraId
            })),
            activePTZCameraId: cameraId
        }));
    },

    updateCameraStatus: (cameraId, status) => set((state) => ({
        ptzCameras: state.ptzCameras.map(cam =>
            cam.id === cameraId ? { ...cam, status } : cam
        ),
        analyticsCameras: state.analyticsCameras.map(cam =>
            cam.id === cameraId ? { ...cam, status } : cam
        )
    })),

    updateAnalyticsCameraHorses: (cameraId, horseIds) => set((state) => ({
        analyticsCameras: state.analyticsCameras.map(cam =>
            cam.id === cameraId ? { ...cam, horsesInView: horseIds } : cam
        )
    })),

    updateCameraActivation: (cameras) => set((state) => ({
        analyticsCameras: state.analyticsCameras.map(cam => ({
            ...cam,
            status: cameras[cam.id] ? 'online' as const : cam.status,
        }))
    })),

    setLiveDetections: (cameras) => set({ liveDetections: cameras }),

    initializeCameras: () => set({
        ptzCameras: createPTZCameras(),
        analyticsCameras: createAnalyticsCameras(),
        activePTZCameraId: getSavedCamera()
    }),

    syncFromStorage: () => {
        const savedCamera = getSavedCamera();
        const currentCamera = get().activePTZCameraId;

        if (savedCamera !== currentCamera) {
            set((state) => ({
                ptzCameras: state.ptzCameras.map(cam => ({
                    ...cam,
                    isActive: cam.id === savedCamera
                })),
                activePTZCameraId: savedCamera
            }));
        }
    }
}));

// Listen for camera changes from other tabs
if (typeof window !== 'undefined') {
    window.addEventListener('storage', (event) => {
        if (event.key === 'activePTZCamera' && event.newValue) {
            useCameraStore.getState().syncFromStorage();
        }
    });
}
