import { create } from 'zustand';
import type { Horse, RaceConfig } from '../types';
import {
    TRACK_LENGTH,
    HORSE_COLORS,
    HORSE_NAMES,
    JOCKEY_NAMES
} from '../types';
import { getDefaultSilkId, getSilkColor } from '../utils/silkUtils';

interface RaceState {
    race: RaceConfig;
    horses: Horse[];
    rankings: Horse[];
    isSimulationRunning: boolean;
    backendConnected: boolean;

    // Actions
    setRace: (race: Partial<RaceConfig>) => void;
    setRaceConfig: (config: { name?: string; totalLaps?: number; trackLength?: number }) => void;
    addHorse: (horse: Partial<Horse> & { id: string; number: number }) => void;
    removeHorse: (horseId: string) => void;
    updateHorse: (horseId: string, updates: Partial<Horse>) => void;
    updateRankings: (horses: Horse[]) => void;
    clearHorses: () => void;
    startRace: () => void;
    stopRace: () => void;
    resetRace: () => void;
    initializeDefaultRace: () => void;
    setBackendConnected: (connected: boolean) => void;
}

// Generate unique ID
const generateId = () => Math.random().toString(36).substr(2, 9);

// Create default horses
const createDefaultHorses = (count: number): Horse[] => {
    return Array.from({ length: count }, (_, i) => {
        const silkId = getDefaultSilkId(i + 1);
        return {
            id: generateId(),
            name: HORSE_NAMES[i % HORSE_NAMES.length],
            number: i + 1,
            color: HORSE_COLORS[i % HORSE_COLORS.length],
            jockeyName: JOCKEY_NAMES[i % JOCKEY_NAMES.length],
            currentPosition: i + 1,
            currentLap: 1,
            timeElapsed: 0,
            gapToLeader: i * 0.5,
            distanceCovered: 0,
            speed: 0,
            lastCameraId: 'cam-0',
            silkId: silkId,
            silkColor: getSilkColor(silkId),
        };
    });
};

// Default race configuration
const defaultRace: RaceConfig = {
    id: generateId(),
    name: 'Grand Championship',
    totalLaps: 1,
    trackLength: TRACK_LENGTH,
    horses: [],
    status: 'pending',
    startTime: null,
    currentTime: 0,
    startFinishPosition: 0, // Default at 0m (top of track)
};

export const useRaceStore = create<RaceState>((set, get) => ({
    race: defaultRace,
    horses: [],
    rankings: [],
    isSimulationRunning: false,
    backendConnected: false,

    setRace: (raceUpdate) => set((state) => ({
        race: { ...state.race, ...raceUpdate }
    })),

    setRaceConfig: (config) => set((state) => ({
        race: {
            ...state.race,
            name: config.name ?? state.race.name,
            totalLaps: config.totalLaps ?? state.race.totalLaps,
            trackLength: config.trackLength ?? state.race.trackLength,
        }
    })),

    addHorse: (horseData) => set((state) => {
        const silkId = horseData.silkId || getDefaultSilkId(horseData.number);
        const horse: Horse = {
            id: horseData.id,
            name: horseData.name || `Horse ${horseData.number}`,
            number: horseData.number,
            color: horseData.color || HORSE_COLORS[(horseData.number - 1) % HORSE_COLORS.length],
            jockeyName: horseData.jockeyName || `Jockey ${horseData.number}`,
            currentPosition: horseData.currentPosition || state.horses.length + 1,
            currentLap: horseData.currentLap || 1,
            timeElapsed: horseData.timeElapsed || 0,
            gapToLeader: horseData.gapToLeader || 0,
            distanceCovered: horseData.distanceCovered || 0,
            speed: horseData.speed || 0,
            lastCameraId: horseData.lastCameraId || 'fixed-0',
            silkId: silkId,
            silkColor: horseData.silkColor || getSilkColor(silkId),
        };
        return {
            horses: [...state.horses, horse],
            rankings: [...state.horses, horse].sort((a, b) => a.currentPosition - b.currentPosition)
        };
    }),

    removeHorse: (horseId) => set((state) => ({
        horses: state.horses.filter(h => h.id !== horseId),
        rankings: state.rankings.filter(h => h.id !== horseId)
    })),

    updateHorse: (horseId, updates) => set((state) => {
        const newHorses = state.horses.map(h =>
            h.id === horseId ? { ...h, ...updates } : h
        );
        return {
            horses: newHorses,
            rankings: [...newHorses].sort((a, b) => a.currentPosition - b.currentPosition)
        };
    }),

    updateRankings: (horses) => set(() => ({
        horses,
        rankings: [...horses].sort((a, b) => a.currentPosition - b.currentPosition)
    })),

    clearHorses: () => set(() => ({
        horses: [],
        rankings: []
    })),

    startRace: () => set((state) => ({
        race: { ...state.race, status: 'active', startTime: Date.now() },
        isSimulationRunning: true
    })),

    stopRace: () => set((state) => ({
        race: { ...state.race, status: 'finished' },
        isSimulationRunning: false
    })),

    resetRace: () => {
        const { race } = get();
        const newHorses = createDefaultHorses(race.horses.length || 10);
        set({
            race: { ...defaultRace, horses: newHorses },
            horses: newHorses,
            rankings: newHorses,
            isSimulationRunning: false
        });
    },

    initializeDefaultRace: () => {
        const horses = createDefaultHorses(5);
        set({
            race: { ...defaultRace, horses },
            horses,
            rankings: horses
        });
    },

    setBackendConnected: (connected) => set(() => ({
        backendConnected: connected
    }))
}));
