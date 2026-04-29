/**
 * Silk (Jokey Kıyafeti) Yardımcı Fonksiyonları
 * Backend'den gelen renk tespiti ile silk eşleştirmesi yapar
 */

import { SILK_COLORS, TOTAL_SILKS } from '../types';

/**
 * Hex renk kodunu RGB değerlerine dönüştürür
 */
export const hexToRgb = (hex: string): { r: number; g: number; b: number } => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
        ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16),
        }
        : { r: 0, g: 0, b: 0 };
};

/**
 * İki renk arasındaki mesafeyi hesaplar (RGB Euclidean distance)
 */
export const colorDistance = (color1: string, color2: string): number => {
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);
    return Math.sqrt(
        Math.pow(rgb1.r - rgb2.r, 2) +
        Math.pow(rgb1.g - rgb2.g, 2) +
        Math.pow(rgb1.b - rgb2.b, 2)
    );
};

/**
 * Tespit edilen renge en yakın silk ID'sini bulur
 * Backend'den gelen renk ile SILK_COLORS arasında eşleştirme yapar
 */
export const findClosestSilkId = (detectedColor: string): number => {
    let closestId = 1;
    let minDistance = Infinity;

    for (let silkId = 1; silkId <= TOTAL_SILKS; silkId++) {
        const silkColor = SILK_COLORS[silkId];
        const distance = colorDistance(detectedColor, silkColor);
        if (distance < minDistance) {
            minDistance = distance;
            closestId = silkId;
        }
    }
    return closestId;
};

/**
 * Silk ID'den görsel dosya yolunu döndürür
 * /assets/silks/silk_1.svg - silk_10.svg (Profesyonel jokey ikonları)
 */
export const getSilkImagePath = (silkId: number): string => {
    const validId = Math.max(1, Math.min(silkId || 1, TOTAL_SILKS));
    return `/assets/silks/silk_${validId}.png`;
};

/**
 * Silk ID'den renk kodunu döndürür
 */
export const getSilkColor = (silkId: number): string => {
    const validId = Math.max(1, Math.min(silkId || 1, TOTAL_SILKS));
    return SILK_COLORS[validId] || SILK_COLORS[1];
};

/**
 * At numarasından varsayılan silk ID hesaplar (döngüsel)
 * At 1-10 → Silk 1-10
 * At 11-20 → Silk 1-10
 * At 21-30 → Silk 1-10
 * ...
 */
export const getDefaultSilkId = (horseNumber: number): number => {
    return ((horseNumber - 1) % TOTAL_SILKS) + 1;
};
