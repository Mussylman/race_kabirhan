#!/usr/bin/env python3
"""Build TRT engine using system TRT via ctypes (no pip tensorrt needed)."""
import ctypes
import subprocess
import sys

# Use system python3 with system TRT libs
code = '''
import ctypes, sys, os

onnx_path = sys.argv[1]
engine_path = sys.argv[2]
fp16 = "--fp16" in sys.argv

# Load system TRT
try:
    trt = ctypes.CDLL("libnvinfer.so.10")
    print(f"Loaded system libnvinfer.so.10")
except:
    print("ERROR: Cannot load system libnvinfer")
    sys.exit(1)

# Use trtexec-like approach via subprocess
# Actually, let's just use the TRT Python bindings from system
import importlib.util
spec = importlib.util.find_spec("tensorrt")
if spec:
    import tensorrt as trt_mod
    print(f"TRT version: {trt_mod.__version__}")
else:
    print("No tensorrt python module, trying LD_PRELOAD approach")
    sys.exit(1)
'''

# Simpler approach: write a C++ builder and compile it
print("Building engine using nvinfer C++ API...")

cpp_code = r'''
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << msg << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: build_engine <onnx> <engine> [--fp16]" << std::endl;
        return 1;
    }
    const char* onnx_path = argv[1];
    const char* engine_path = argv[2];
    bool fp16 = false;
    for (int i = 3; i < argc; i++)
        if (strcmp(argv[i], "--fp16") == 0) fp16 = true;

    Logger logger;
    auto builder = nvinfer1::createInferBuilder(logger);
    auto network = builder->createNetworkV2(
        1u << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto parser = nvonnxparser::createParser(*network, logger);

    if (!parser->parseFromFile(onnx_path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX" << std::endl;
        return 1;
    }

    auto config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Dynamic batch
    auto profile = builder->createOptimizationProfile();
    auto input = network->getInput(0);
    auto dims = input->getDimensions();
    std::cerr << "Input: " << input->getName() << " [" << dims.d[0] << "," << dims.d[1]
              << "," << dims.d[2] << "," << dims.d[3] << "]" << std::endl;

    nvinfer1::Dims4 minD(1, dims.d[1], dims.d[2], dims.d[3]);
    nvinfer1::Dims4 optD(16, dims.d[1], dims.d[2], dims.d[3]);
    nvinfer1::Dims4 maxD(128, dims.d[1], dims.d[2], dims.d[3]);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minD);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optD);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxD);
    config->addOptimizationProfile(profile);

    std::cerr << "Building engine" << (fp16 ? " (FP16)" : " (FP32)") << "..." << std::endl;
    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) { std::cerr << "Build failed" << std::endl; return 1; }

    std::ofstream out(engine_path, std::ios::binary);
    out.write(static_cast<const char*>(plan->data()), plan->size());
    out.close();

    std::cerr << "Saved: " << engine_path << " (" << plan->size() / 1000000 << " MB)" << std::endl;

    delete plan; delete config; delete parser; delete network; delete builder;
    return 0;
}
'''

if __name__ == "__main__":
    import tempfile, os

    src = "/tmp/build_engine.cpp"
    binary = "/tmp/build_engine"

    with open(src, 'w') as f:
        f.write(cpp_code)

    # Compile with system TRT
    cmd = [
        "g++", "-O2", src, "-o", binary,
        "-I/usr/include/x86_64-linux-gnu",
        "-lnvinfer", "-lnvonnxparser",
        "-Wl,-rpath,/lib/x86_64-linux-gnu"
    ]
    print(f"Compiling: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Compile error:\n{r.stderr}")
        sys.exit(1)

    # Run with args
    args = sys.argv[1:]
    if not args:
        print(f"Usage: python build_engine.py <onnx> <engine> [--fp16]")
        sys.exit(0)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    r = subprocess.run([binary] + args, env=env)
    sys.exit(r.returncode)
