/**
 * diag_logger.cpp — CSV + JPG diagnostic logger for DeepStream pipeline.
 */

#include "diag_logger.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>

#include <cuda_runtime.h>
#include "nvbufsurface.h"

// stb_image_write for JPG saving (header-only)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

namespace rv {

DiagLogger::~DiagLogger() {
    if (enabled_ && total_dets_ > 0) {
        fprintf(stderr, "\n[DiagLogger] ═══ CROP SIZE STATISTICS ═══\n");
        fprintf(stderr, "[DiagLogger] Total detections: %d\n", total_dets_);
        fprintf(stderr, "[DiagLogger] Crop width:  min=%d  avg=%lld  max=%d\n",
                min_crop_w_, sum_crop_w_ / total_dets_, max_crop_w_);
        fprintf(stderr, "[DiagLogger] Crop height: min=%d  avg=%lld  max=%d\n",
                min_crop_h_, sum_crop_h_ / total_dets_, max_crop_h_);
        fprintf(stderr, "[DiagLogger] Avg crop area: %lld px\n",
                (sum_crop_w_ / total_dets_) * (sum_crop_h_ / total_dets_));
        fprintf(stderr, "[DiagLogger] Log dir: %s\n", exp_dir_.c_str());
        fprintf(stderr, "[DiagLogger] ═════════════════════════════\n\n");
    }
    if (csv_file_) {
        fclose(csv_file_);
        csv_file_ = nullptr;
    }
}

bool DiagLogger::init(const std::string& base_dir, int snap_interval) {
    snap_interval_ = snap_interval;

    // Create base dir
    fs::create_directories(base_dir);

    // Find next exp number
    int max_exp = 0;
    for (const auto& entry : fs::directory_iterator(base_dir)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            if (name.substr(0, 3) == "exp") {
                try {
                    int n = std::stoi(name.substr(3));
                    max_exp = std::max(max_exp, n);
                } catch (...) {}
            }
        }
    }

    exp_dir_ = base_dir + "/exp" + std::to_string(max_exp + 1);
    fs::create_directories(exp_dir_);
    fs::create_directories(exp_dir_ + "/frames");

    // Open CSV
    std::string csv_path = exp_dir_ + "/detections.csv";
    csv_file_ = fopen(csv_path.c_str(), "w");
    if (!csv_file_) {
        fprintf(stderr, "[DiagLogger] Cannot create CSV: %s\n", csv_path.c_str());
        return false;
    }

    // Write header
    fprintf(csv_file_,
        "batch,cam_id,frame_w,frame_h,det_idx,"
        "x1,y1,x2,y2,center_x,"
        "crop_w,crop_h,crop_pixels,"
        "det_conf,color,color_conf,"
        "prob_blue,prob_green,prob_purple,prob_red,prob_yellow,"
        "track_id\n");
    fflush(csv_file_);

    enabled_ = true;
    fprintf(stderr, "[DiagLogger] Logging to %s (snap every %d batches)\n",
            exp_dir_.c_str(), snap_interval_);
    return true;
}

void DiagLogger::log_detections(int batch_num,
                                const std::vector<DiagDetection>& detections) {
    if (!enabled_ || !csv_file_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& d : detections) {
        const char* color_name = (d.color_id < NUM_COLORS)
            ? COLOR_NAMES[d.color_id] : "unknown";

        int crop_w = static_cast<int>(d.x2 - d.x1);
        int crop_h = static_cast<int>(d.y2 - d.y1);
        int crop_px = crop_w * crop_h;

        // Accumulate crop size stats
        total_dets_++;
        sum_crop_w_ += crop_w;
        sum_crop_h_ += crop_h;
        if (crop_w < min_crop_w_) min_crop_w_ = crop_w;
        if (crop_h < min_crop_h_) min_crop_h_ = crop_h;
        if (crop_w > max_crop_w_) max_crop_w_ = crop_w;
        if (crop_h > max_crop_h_) max_crop_h_ = crop_h;

        fprintf(csv_file_,
            "%d,%s,%d,%d,%d,"
            "%.1f,%.1f,%.1f,%.1f,%.1f,"
            "%d,%d,%d,"
            "%.4f,%s,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,%.4f,"
            "%u\n",
            batch_num, d.cam_id.c_str(), d.frame_w, d.frame_h, d.det_idx,
            d.x1, d.y1, d.x2, d.y2, d.center_x,
            crop_w, crop_h, crop_px,
            d.det_conf, color_name, d.color_conf,
            d.color_probs[COLOR_BLUE], d.color_probs[COLOR_GREEN],
            d.color_probs[COLOR_PURPLE], d.color_probs[COLOR_RED],
            d.color_probs[COLOR_YELLOW],
            d.track_id);
    }
    fflush(csv_file_);
}

// ── Simple in-memory drawing (no OpenCV dependency) ─────────────────

static void draw_rect(uint8_t* img, int w, int h, int channels,
                       int x1, int y1, int x2, int y2,
                       uint8_t r, uint8_t g, uint8_t b, int thickness) {
    // Clamp
    x1 = std::max(0, std::min(x1, w - 1));
    y1 = std::max(0, std::min(y1, h - 1));
    x2 = std::max(0, std::min(x2, w - 1));
    y2 = std::max(0, std::min(y2, h - 1));

    for (int t = 0; t < thickness; ++t) {
        int cx1 = x1 + t, cy1 = y1 + t, cx2 = x2 - t, cy2 = y2 - t;
        if (cx1 >= cx2 || cy1 >= cy2) break;

        // Top & bottom
        for (int x = cx1; x <= cx2; ++x) {
            int idx_top = (cy1 * w + x) * channels;
            int idx_bot = (cy2 * w + x) * channels;
            img[idx_top] = r; img[idx_top+1] = g; img[idx_top+2] = b;
            img[idx_bot] = r; img[idx_bot+1] = g; img[idx_bot+2] = b;
        }
        // Left & right
        for (int y = cy1; y <= cy2; ++y) {
            int idx_left  = (y * w + cx1) * channels;
            int idx_right = (y * w + cx2) * channels;
            img[idx_left]  = r; img[idx_left+1]  = g; img[idx_left+2]  = b;
            img[idx_right] = r; img[idx_right+1] = g; img[idx_right+2] = b;
        }
    }
}

// Simple 5x7 bitmap font for labels
static const uint8_t FONT_5X7[][7] = {
    // space, 0-9, A-Z, %, .
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space (0)
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0 (1)
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1 (2)
    {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}, // 2 (3)
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, // 3 (4)
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4 (5)
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5 (6)
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6 (7)
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7 (8)
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8 (9)
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9 (10)
    // A-Z (11-36)
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // A
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C
    {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, // D
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // E
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // F
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, // G
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, // H
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, // I
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, // J
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // K
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, // L
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, // M
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // N
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // O
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, // P
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, // Q
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // R
    {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E}, // S
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // T
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, // U
    {0x11,0x11,0x11,0x11,0x0A,0x0A,0x04}, // V
    {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}, // W
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, // X
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, // Y
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // Z
    {0x12,0x05,0x02,0x04,0x0A,0x14,0x09}, // % (37)
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, // . (38)
    {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, // + (39) unused
};

static int char_index(char c) {
    if (c == ' ') return 0;
    if (c >= '0' && c <= '9') return 1 + (c - '0');
    if (c >= 'A' && c <= 'Z') return 11 + (c - 'A');
    if (c >= 'a' && c <= 'z') return 11 + (c - 'a');
    if (c == '%') return 37;
    if (c == '.') return 38;
    if (c == '-') return 0; // treat as space
    if (c == '#') return 0;
    return 0;
}

static void draw_text(uint8_t* img, int w, int h, int channels,
                       int px, int py, const char* text,
                       uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
    int cursor_x = px;
    for (const char* p = text; *p; ++p) {
        int ci = char_index(*p);
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = FONT_5X7[ci][row];
            for (int col = 0; col < 5; ++col) {
                if (bits & (0x10 >> col)) {
                    for (int sy = 0; sy < scale; ++sy) {
                        for (int sx = 0; sx < scale; ++sx) {
                            int xx = cursor_x + col * scale + sx;
                            int yy = py + row * scale + sy;
                            if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
                                int idx = (yy * w + xx) * channels;
                                img[idx] = r; img[idx+1] = g; img[idx+2] = b;
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

// ── NV12 → RGB conversion helper ────────────────────────────────────
static inline uint8_t clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static void nv12_to_rgb(const uint8_t* y_plane, int y_pitch,
                          const uint8_t* uv_plane, int uv_pitch,
                          uint8_t* rgb, int w, int h) {
    for (int row = 0; row < h; ++row) {
        const uint8_t* y_row  = y_plane  + row * y_pitch;
        const uint8_t* uv_row = uv_plane + (row / 2) * uv_pitch;
        for (int col = 0; col < w; ++col) {
            int Y = y_row[col];
            int U = uv_row[(col / 2) * 2]     - 128;
            int V = uv_row[(col / 2) * 2 + 1] - 128;

            int R = Y + ((359 * V) >> 8);
            int G = Y - ((88 * U + 183 * V) >> 8);
            int B = Y + ((454 * U) >> 8);

            int idx = (row * w + col) * 3;
            rgb[idx + 0] = clamp_u8(R);
            rgb[idx + 1] = clamp_u8(G);
            rgb[idx + 2] = clamp_u8(B);
        }
    }
}

void DiagLogger::save_snapshot(int batch_num, const NvBufSurface* surface,
                                int cam_index, const std::string& cam_id,
                                const std::vector<DiagDetection>& cam_dets) {
    if (!enabled_ || !surface) return;

    if (cam_index >= static_cast<int>(surface->numFilled)) return;

    const auto& params = surface->surfaceList[cam_index];
    int fw = params.width;
    int fh = params.height;

    NvBufSurface* surf = const_cast<NvBufSurface*>(surface);
    bool did_map = false;

    // Determine surface color format
    NvBufSurfaceColorFormat fmt = params.colorFormat;
    bool is_nv12 = (fmt == NVBUF_COLOR_FORMAT_NV12 || fmt == NVBUF_COLOR_FORMAT_NV12_709 ||
                    fmt == NVBUF_COLOR_FORMAT_NV12_709_ER ||
                    fmt == NVBUF_COLOR_FORMAT_NV12_2020 ||
                    fmt == NVBUF_COLOR_FORMAT_NV12_ER);
    bool is_rgba = (fmt == NVBUF_COLOR_FORMAT_RGBA);

    if (!is_nv12 && !is_rgba) {
        fprintf(stderr, "[DiagLogger] Unsupported surface format %d for %s, skipping\n",
                static_cast<int>(fmt), cam_id.c_str());
        return;
    }

    std::vector<uint8_t> rgb(fw * fh * 3);

    if (is_nv12) {
        // NV12: Y plane (fw*fh) + UV interleaved plane (fw * fh/2)
        int y_pitch = params.pitch;
        int uv_pitch = params.pitch;
        size_t y_plane_sz = y_pitch * fh;

        std::vector<uint8_t> y_data(y_pitch * fh);
        std::vector<uint8_t> uv_data(uv_pitch * (fh / 2));

        if (params.mappedAddr.addr[0]) {
            // Already mapped — Y plane at addr[0], UV at addr[1] or offset
            const uint8_t* base = static_cast<const uint8_t*>(params.mappedAddr.addr[0]);
            memcpy(y_data.data(), base, y_pitch * fh);
            if (params.mappedAddr.addr[1]) {
                memcpy(uv_data.data(), params.mappedAddr.addr[1], uv_pitch * (fh / 2));
            } else {
                memcpy(uv_data.data(), base + y_plane_sz, uv_pitch * (fh / 2));
            }
        } else if (surf->memType == NVBUF_MEM_CUDA_DEVICE ||
                   surf->memType == NVBUF_MEM_CUDA_UNIFIED) {
            const uint8_t* gpu_base = static_cast<const uint8_t*>(params.dataPtr);
            // Y plane
            cudaMemcpy2D(y_data.data(), y_pitch,
                          gpu_base, y_pitch,
                          fw, fh,
                          cudaMemcpyDeviceToHost);
            // UV plane (right after Y)
            cudaMemcpy2D(uv_data.data(), uv_pitch,
                          gpu_base + y_plane_sz, uv_pitch,
                          fw, fh / 2,
                          cudaMemcpyDeviceToHost);
        } else {
            if (NvBufSurfaceMap(surf, cam_index, -1, NVBUF_MAP_READ) == 0) {
                did_map = true;
                NvBufSurfaceSyncForCpu(surf, cam_index, -1);
                const uint8_t* y_ptr = static_cast<const uint8_t*>(
                    surf->surfaceList[cam_index].mappedAddr.addr[0]);
                memcpy(y_data.data(), y_ptr, y_pitch * fh);
                if (surf->surfaceList[cam_index].mappedAddr.addr[1]) {
                    memcpy(uv_data.data(),
                           surf->surfaceList[cam_index].mappedAddr.addr[1],
                           uv_pitch * (fh / 2));
                } else {
                    memcpy(uv_data.data(), y_ptr + y_plane_sz, uv_pitch * (fh / 2));
                }
            } else {
                fprintf(stderr, "[DiagLogger] Cannot map NV12 surface for %s\n", cam_id.c_str());
                return;
            }
        }

        nv12_to_rgb(y_data.data(), y_pitch, uv_data.data(), uv_pitch,
                     rgb.data(), fw, fh);
    } else {
        // RGBA path (original)
        std::vector<uint8_t> h_frame(fw * fh * 4);

        if (params.mappedAddr.addr[0]) {
            memcpy(h_frame.data(), params.mappedAddr.addr[0], fw * fh * 4);
        } else if (surf->memType == NVBUF_MEM_CUDA_DEVICE ||
                   surf->memType == NVBUF_MEM_CUDA_UNIFIED) {
            cudaMemcpy2D(h_frame.data(), fw * 4,
                          params.dataPtr, params.pitch,
                          fw * 4, fh,
                          cudaMemcpyDeviceToHost);
        } else {
            if (NvBufSurfaceMap(surf, cam_index, 0, NVBUF_MAP_READ) == 0) {
                did_map = true;
                NvBufSurfaceSyncForCpu(surf, cam_index, 0);
                memcpy(h_frame.data(),
                       surf->surfaceList[cam_index].mappedAddr.addr[0],
                       fw * fh * 4);
            } else {
                fprintf(stderr, "[DiagLogger] Cannot map RGBA surface for %s\n", cam_id.c_str());
                return;
            }
        }

        for (int i = 0; i < fw * fh; ++i) {
            rgb[i * 3 + 0] = h_frame[i * 4 + 0];
            rgb[i * 3 + 1] = h_frame[i * 4 + 1];
            rgb[i * 3 + 2] = h_frame[i * 4 + 2];
        }
    }

    // Color map for drawing
    static const uint8_t BOX_COLORS[][3] = {
        {0, 100, 255},   // blue
        {0, 230, 0},     // green
        {200, 0, 200},   // purple
        {255, 0, 0},     // red
        {255, 230, 0},   // yellow
    };

    // Draw bboxes + labels
    for (const auto& d : cam_dets) {
        uint8_t br = 255, bg = 255, bb = 255; // white default
        if (d.color_id < NUM_COLORS) {
            br = BOX_COLORS[d.color_id][0];
            bg = BOX_COLORS[d.color_id][1];
            bb = BOX_COLORS[d.color_id][2];
        }

        draw_rect(rgb.data(), fw, fh, 3,
                   static_cast<int>(d.x1), static_cast<int>(d.y1),
                   static_cast<int>(d.x2), static_cast<int>(d.y2),
                   br, bg, bb, 3);

        // Label: "COLOR CONF%"
        char label[64];
        const char* cname = (d.color_id < NUM_COLORS)
            ? COLOR_NAMES[d.color_id] : "UNK";
        snprintf(label, sizeof(label), "%s %d%%", cname,
                 static_cast<int>(d.color_conf * 100));

        draw_text(rgb.data(), fw, fh, 3,
                   static_cast<int>(d.x1), std::max(0, static_cast<int>(d.y1) - 18),
                   label, br, bg, bb, 2);

        // Crop region = full bbox now (no torso sub-crop)
        // Draw crop size label below bbox
        int crop_w = static_cast<int>(d.x2 - d.x1);
        int crop_h = static_cast<int>(d.y2 - d.y1);
        char size_label[32];
        snprintf(size_label, sizeof(size_label), "%dx%d", crop_w, crop_h);
        draw_text(rgb.data(), fw, fh, 3,
                   static_cast<int>(d.x1), static_cast<int>(d.y2) + 2,
                   size_label, 255, 255, 0, 1);
    }

    // Save individual crop images (before drawing labels on full frame)
    {
        static bool crops_dir_created = false;
        if (!crops_dir_created) {
            fs::create_directories(exp_dir_ + "/crops");
            crops_dir_created = true;
        }
        for (const auto& d : cam_dets) {
            int cx1 = std::max(0, static_cast<int>(d.x1));
            int cy1 = std::max(0, static_cast<int>(d.y1));
            int cx2 = std::min(fw, static_cast<int>(d.x2));
            int cy2 = std::min(fh, static_cast<int>(d.y2));
            int cw = cx2 - cx1;
            int ch = cy2 - cy1;
            if (cw > 2 && ch > 2) {
                std::vector<uint8_t> crop_rgb(cw * ch * 3);
                for (int row = 0; row < ch; ++row) {
                    memcpy(crop_rgb.data() + row * cw * 3,
                           rgb.data() + ((cy1 + row) * fw + cx1) * 3,
                           cw * 3);
                }
                const char* cname = (d.color_id < NUM_COLORS)
                    ? COLOR_NAMES[d.color_id] : "unk";
                char crop_fn[256];
                snprintf(crop_fn, sizeof(crop_fn),
                         "%s/crops/b%05d_%s_t%u_%s_%dx%d.jpg",
                         exp_dir_.c_str(), batch_num, cam_id.c_str(),
                         d.track_id, cname, cw, ch);
                stbi_write_jpg(crop_fn, cw, ch, 3, crop_rgb.data(), 95);
            }
        }
    }

    // Draw camera label
    char cam_label[64];
    snprintf(cam_label, sizeof(cam_label), "%s B%d D%d",
             cam_id.c_str(), batch_num, static_cast<int>(cam_dets.size()));
    draw_text(rgb.data(), fw, fh, 3, 10, 10, cam_label, 255, 255, 255, 3);

    // Save JPG
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/frames/%s_batch%05d.jpg",
             exp_dir_.c_str(), cam_id.c_str(), batch_num);

    stbi_write_jpg(filename, fw, fh, 3, rgb.data(), 90);

    // Unmap if we mapped
    if (did_map) {
        NvBufSurfaceUnMap(surf, cam_index, 0);
    }
}

} // namespace rv
