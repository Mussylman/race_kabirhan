# ============================================================
# Race Vision — Unified Backend (Python API + DeepStream C++)
#
# Single image: Python launches C++ DeepStream as subprocess.
# No torch/ultralytics — all inference done in C++ via TensorRT.
# ============================================================

# ── Build stage: compile C++ DeepStream binary ────────────────
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential pkg-config \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libglib2.0-dev nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY deepstream/src/ src/
COPY deepstream/CMakeLists.txt .
COPY deepstream/configs/ configs/

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DDEEPSTREAM_SDK=/opt/nvidia/deepstream/deepstream \
    && make -j$(nproc)

# ── Runtime stage ─────────────────────────────────────────────
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Lightweight Python deps (no torch/ultralytics)
COPY requirements_deepstream.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements_deepstream.txt

WORKDIR /app

# C++ binary + custom parser library
COPY --from=builder /build/build/race_vision_deepstream /app/bin/
COPY --from=builder /build/build/libnvdsinfer_yolov8_parser.so /app/lib/
COPY deepstream/configs/ /app/deepstream_configs/

# Python code
COPY api/ api/
COPY pipeline/ pipeline/
COPY tools/ tools/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/stats')" || exit 1

ENTRYPOINT ["python3", "-m", "api.server"]
CMD ["--config", "cameras_example.json", "--deepstream", "--host", "0.0.0.0", "--port", "8000"]
