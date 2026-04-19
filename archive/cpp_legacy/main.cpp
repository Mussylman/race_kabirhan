/**
 * main.cpp — DeepStream Race Vision entry point.
 *
 * Parses cameras.json config, builds the GStreamer pipeline, and runs
 * the GLib main loop. Detection results are written to POSIX shared memory
 * for consumption by the Python FastAPI server.
 *
 * Usage:
 *   # Single pipeline (all cameras through YOLOv8s):
 *   ./race_vision_deepstream --config cameras.json \
 *       --yolo-engine models/yolov8s_deepstream.engine \
 *       --color-engine models/color_classifier.engine
 *
 *   # Dual pipeline (trigger YOLOv8n + analysis YOLOv8s):
 *   ./race_vision_deepstream --dual --config cameras.json \
 *       --trigger-conf configs/nvinfer_yolov8n_trigger.txt \
 *       --yolo-engine configs/nvinfer_yolov8s_analysis.txt \
 *       --color-engine models/color_classifier.engine
 */

#include "pipeline.h"
#include "dual_pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static rv::Pipeline*     g_pipeline      = nullptr;
static rv::DualPipeline* g_dual_pipeline = nullptr;

static void signal_handler(int sig) {
    fprintf(stderr, "\n[Main] Caught signal %d, stopping...\n", sig);
    if (g_pipeline) {
        g_pipeline->stop();
    }
    if (g_dual_pipeline) {
        g_dual_pipeline->stop();
    }
}

static std::vector<rv::CameraConfig> load_cameras(const std::string& config_path) {
    std::vector<rv::CameraConfig> cameras;

    std::ifstream f(config_path);
    if (!f.good()) {
        fprintf(stderr, "[Main] Cannot open config: %s\n", config_path.c_str());
        return cameras;
    }

    json j = json::parse(f);

    if (j.contains("analytics")) {
        for (const auto& cam : j["analytics"]) {
            rv::CameraConfig cc;
            cc.id  = cam.value("id", "");
            cc.url = cam.value("url", "");
            cc.track_start = cam.value("track_start", 0.0f);
            cc.track_end   = cam.value("track_end", 100.0f);
            cameras.push_back(cc);
        }
    }

    fprintf(stderr, "[Main] Loaded %zu cameras from %s\n",
            cameras.size(), config_path.c_str());

    // Load ROI zones (normalized coordinates)
    std::string roi_path = config_path;
    auto slash = roi_path.rfind('/');
    if (slash != std::string::npos)
        roi_path = roi_path.substr(0, slash + 1) + "camera_roi_normalized.json";
    else
        roi_path = "configs/camera_roi_normalized.json";

    std::ifstream froi(roi_path);
    if (froi.good()) {
        json jroi = json::parse(froi);
        int roi_count = 0;
        for (auto& cc : cameras) {
            if (jroi.contains(cc.id)) {
                for (const auto& zone : jroi[cc.id]) {
                    std::vector<rv::Point2f> poly;
                    for (const auto& pt : zone) {
                        poly.push_back({pt.value("x", 0.0f), pt.value("y", 0.0f)});
                    }
                    cc.roi_zones.push_back(poly);
                    roi_count++;
                }
            }
        }
        fprintf(stderr, "[Main] Loaded %d ROI zones from %s\n", roi_count, roi_path.c_str());
    }

    return cameras;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\nSingle pipeline (default):\n");
    fprintf(stderr, "  --config <path>        Camera config JSON (default: cameras_example.json)\n");
    fprintf(stderr, "  --yolo-engine <path>   YOLOv8s TRT engine or nvinfer config .txt\n");
    fprintf(stderr, "  --color-engine <path>  Color classifier TRT engine\n");
    fprintf(stderr, "  --mux-width <int>      Streammux width (default: 1280)\n");
    fprintf(stderr, "  --mux-height <int>     Streammux height (default: 1280)\n");
    fprintf(stderr, "  --conf <float>         Detection confidence threshold (default: 0.35)\n");
    fprintf(stderr, "  --file-mode            Use file:// URIs (live-source always TRUE)\n");
    fprintf(stderr, "  --display              Show video grid with OSD (requires X11)\n");
    fprintf(stderr, "\nDual pipeline (trigger + analysis):\n");
    fprintf(stderr, "  --dual                 Enable dual-pipeline mode\n");
    fprintf(stderr, "  --trigger-conf <path>  Trigger nvinfer config (default: configs/nvinfer_yolov8n_trigger.txt)\n");
    fprintf(stderr, "  --cooldown <float>     Trigger cooldown seconds (default: 3.0)\n");
    fprintf(stderr, "  --max-active <int>     Max active analysis cameras (default: 8)\n");
    fprintf(stderr, "\nDiagnostics:\n");
    fprintf(stderr, "  --log-dir <path>       Save CSV + JPG snapshots (auto expN/ dirs)\n");
    fprintf(stderr, "  --snap-interval <int>  Save JPG every N batches (default: 10, 0=all)\n");
    fprintf(stderr, "\n  --help                 Show this help\n");
}

int main(int argc, char* argv[]) {
    std::string config_path   = "cameras_example.json";
    std::string yolo_engine   = "models/yolov8s_deepstream.engine";
    std::string color_engine  = "models/color_classifier.engine";
    // Keep at 800x800 to match existing TRT engine
    int mux_width  = 800;
    int mux_height = 800;
    float det_conf = 0.35f;

    // Dual pipeline options
    bool dual_mode = false;
    bool file_mode = false;   // --file-mode: set live-source=FALSE for file:// URIs
    bool display_mode = false;      // --display: show video with OSD + tiler
    bool display_only_mode = false; // --display-only: video grid, no inference
    std::string trigger_conf = "configs/nvinfer_yolov8n_trigger.txt";
    float cooldown = 3.0f;
    int max_active = 8;

    // Diagnostic logging
    std::string log_dir;        // --log-dir: save CSV + JPG to ds_results/expN/
    int snap_interval = 10;     // --snap-interval: save JPG every N batches

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--yolo-engine" && i + 1 < argc) {
            yolo_engine = argv[++i];
        } else if (arg == "--color-engine" && i + 1 < argc) {
            color_engine = argv[++i];
        } else if (arg == "--mux-width" && i + 1 < argc) {
            mux_width = std::atoi(argv[++i]);
        } else if (arg == "--mux-height" && i + 1 < argc) {
            mux_height = std::atoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            det_conf = std::atof(argv[++i]);
        } else if (arg == "--dual") {
            dual_mode = true;
        } else if (arg == "--file-mode") {
            file_mode = true;
        } else if (arg == "--display") {
            display_mode = true;
        } else if (arg == "--display-only") {
            display_mode = true;
            display_only_mode = true;
        } else if (arg == "--trigger-conf" && i + 1 < argc) {
            trigger_conf = argv[++i];
        } else if (arg == "--cooldown" && i + 1 < argc) {
            cooldown = std::atof(argv[++i]);
        } else if (arg == "--max-active" && i + 1 < argc) {
            max_active = std::atoi(argv[++i]);
        } else if (arg == "--log-dir" && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (arg == "--snap-interval" && i + 1 < argc) {
            snap_interval = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[Main] Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Signal handlers
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // Load cameras
    auto cameras = load_cameras(config_path);
    if (cameras.empty()) {
        fprintf(stderr, "[Main] No cameras configured, exiting\n");
        return 1;
    }

    if (dual_mode) {
        // ── Dual pipeline mode ─────────────────────────────────────
        fprintf(stderr, "[Main] Dual pipeline mode (trigger + analysis)\n");

        rv::DualPipelineConfig dual_config;

        // Trigger config
        dual_config.trigger.cameras       = cameras;
        dual_config.trigger.nvinfer_config = trigger_conf;
        dual_config.trigger.mux_width     = 640;
        dual_config.trigger.mux_height    = 640;
        dual_config.trigger.max_active    = max_active;
        dual_config.trigger.cooldown_s    = cooldown;

        // Analysis config
        dual_config.analysis.cameras          = cameras;
        dual_config.analysis.nvinfer_config   = yolo_engine;
        dual_config.analysis.color_engine_path = color_engine;
        dual_config.analysis.mux_width        = mux_width;
        dual_config.analysis.mux_height       = mux_height;
        dual_config.analysis.max_batch        = max_active;

        rv::DualPipeline dual;
        g_dual_pipeline = &dual;

        if (!dual.build(dual_config)) {
            fprintf(stderr, "[Main] Failed to build dual pipeline\n");
            return 1;
        }

        dual.start();

        fprintf(stderr, "[Main] Running dual pipeline (Ctrl+C to stop)...\n");
        fprintf(stderr, "[Main] Trigger SHM: %s  Analysis SHM: %s\n",
                rv::TRIGGER_SHM_NAME, rv::SHM_NAME);
        fprintf(stderr, "[Main] Max active: %d  Cooldown: %.1fs\n",
                max_active, cooldown);

        GMainLoop* loop = dual.get_main_loop();
        if (loop) {
            g_main_loop_run(loop);
        }

        dual.stop();
        g_dual_pipeline = nullptr;

    } else {
        // ── Single pipeline mode (unchanged) ───────────────────────
        rv::PipelineConfig pipeline_config;
        pipeline_config.cameras          = cameras;
        pipeline_config.yolo_engine_path = yolo_engine;
        pipeline_config.color_engine_path = color_engine;
        pipeline_config.mux_width        = mux_width;
        pipeline_config.mux_height       = mux_height;
        pipeline_config.det_conf         = det_conf;
        // live-source=FALSE enables proper sync/clock for file playback
        // live-source=TRUE required for >7 file sources (FALSE deadlocks waiting for all decoders)
        pipeline_config.live_source      = !file_mode || (cameras.size() > 7);
        pipeline_config.display          = display_mode;
        pipeline_config.display_only     = display_only_mode;
        pipeline_config.log_dir          = log_dir;
        pipeline_config.snap_interval    = snap_interval;
        if (file_mode) {
            pipeline_config.mux_batched_push_timeout = 1000000;  // 1s for file decode
        } else {
            pipeline_config.mux_batched_push_timeout = 40000;   // 40ms for live RTSP
        }
        pipeline_config.batch_size       = static_cast<int>(cameras.size());
        if (pipeline_config.batch_size > rv::MAX_CAMERAS) {
            pipeline_config.batch_size = rv::MAX_CAMERAS;
        }

        rv::Pipeline pipeline;
        g_pipeline = &pipeline;

        if (!pipeline.build(pipeline_config)) {
            fprintf(stderr, "[Main] Failed to build pipeline\n");
            return 1;
        }

        pipeline.start();

        fprintf(stderr, "[Main] Running (Ctrl+C to stop)...\n");
        fprintf(stderr, "[Main] SHM: %s  SEM: %s\n", rv::SHM_NAME, rv::SEM_NAME);

        GMainLoop* loop = pipeline.get_main_loop();
        if (loop) {
            g_main_loop_run(loop);
        }

        pipeline.stop();
        g_pipeline = nullptr;
    }

    fprintf(stderr, "[Main] Exited cleanly\n");
    return 0;
}
