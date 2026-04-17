#!/bin/bash
# ================================================================
# Race Vision — Полная установка на RTX 5070 Ti (Blackwell)
# ================================================================
#
# Сервер: Intel i7-14700F, RTX 5070 Ti 16GB, 32GB DDR5
# ОС:     Ubuntu 24.04 LTS
#
# Порядок:
#   1. NVIDIA Driver 570+
#   2. CUDA 12.8
#   3. cuDNN 9.x
#   4. TensorRT 10.6
#   5. GStreamer
#   6. DeepStream 9.0
#   7. Python 3.12 + venv
#   8. PyTorch nightly (sm_120)
#   9. Ultralytics + ONNX + остальное
#  10. Сборка C++ плагина
#  11. Проверка всего стека
#
# Время: ~1 час (без скачивания)
#
# ВАЖНО: Выполнять по блокам, после каждого — проверка.
#        Не запускать весь скрипт целиком.
# ================================================================

set -e

echo "============================================"
echo "  Race Vision — установка зависимостей"
echo "============================================"

# ================================================================
# 0. ПОДГОТОВКА СИСТЕМЫ
# ================================================================

sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    dkms \
    git \
    curl \
    wget \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libtbb-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libjson-glib-dev

echo "[OK] Системные пакеты установлены"

# ================================================================
# 1. NVIDIA DRIVER 570+
# ================================================================
# RTX 5070 Ti (Blackwell) требует минимум driver 570.
# Рекомендуется 570-open (open-source kernel module).

echo ""
echo "=== 1. NVIDIA Driver ==="

# Удалить старые драйверы (если есть)
sudo apt remove --purge -y 'nvidia-*' 2>/dev/null || true
sudo apt autoremove -y

# Добавить PPA с драйверами
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update

# Установить driver
sudo apt install -y nvidia-driver-570-open

echo ""
echo "[!] НУЖНА ПЕРЕЗАГРУЗКА после установки драйвера."
echo "    Выполни: sudo reboot"
echo "    После перезагрузки проверь: nvidia-smi"
echo ""
echo "    Ожидаемый результат:"
echo "    +-----------------------------------------------+"
echo "    | NVIDIA-SMI 572.xx   Driver Version: 572.xx    |"
echo "    | CUDA Version: 13.x                            |"
echo "    | GPU: NVIDIA GeForce RTX 5070 Ti               |"
echo "    +-----------------------------------------------+"
echo ""
read -p "Перезагрузился и nvidia-smi работает? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Сначала перезагрузись: sudo reboot"
    exit 1
fi

# ================================================================
# 2. CUDA 12.8
# ================================================================
# Выбираем 12.8 а не 13.2, потому что:
# - PyTorch stable/nightly собран под cu12
# - Полная поддержка Blackwell через PTX (быстрый JIT)
# - Все библиотеки совместимы
# Когда PyTorch 2.7 выйдет — можно обновиться до CUDA 13.2

echo ""
echo "=== 2. CUDA Toolkit 12.8 ==="

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update

sudo apt install -y cuda-toolkit-12-8

# Прописать PATH
CUDA_PROFILE="export PATH=/usr/local/cuda-12.8/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH"

if ! grep -q "cuda-12.8" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA 12.8" >> ~/.bashrc
    echo "$CUDA_PROFILE" >> ~/.bashrc
fi
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Проверка
echo ""
echo "Проверка CUDA:"
nvcc --version
echo ""

# ================================================================
# 3. cuDNN 9.x
# ================================================================
# cuDNN — оптимизированные операции для нейросетей (свёртки, attention).
# Версия 9.x поддерживает CUDA 12.x и Blackwell.

echo ""
echo "=== 3. cuDNN ==="

sudo apt install -y cudnn9-cuda-12

# Проверка
echo ""
echo "Проверка cuDNN:"
dpkg -l | grep cudnn
echo ""

# ================================================================
# 4. TensorRT 10.6
# ================================================================
# TensorRT компилирует ONNX модели в оптимизированные engine
# под конкретный GPU. Ядро DeepStream inference.

echo ""
echo "=== 4. TensorRT ==="

sudo apt install -y tensorrt

# Также ставим Python bindings
pip3 install tensorrt 2>/dev/null || echo "TRT Python bindings — установим позже в venv"

# Проверка
echo ""
echo "Проверка TensorRT:"
dpkg -l | grep tensorrt | head -3
echo ""

# ================================================================
# 5. GStreamer
# ================================================================
# GStreamer — мультимедийный фреймворк, основа DeepStream.
# Нужны base + good + bad + ugly плагины + RTSP сервер.

echo ""
echo "=== 5. GStreamer ==="

sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    libgstrtspserver-1.0-dev

# Проверка
echo ""
echo "Проверка GStreamer:"
gst-inspect-1.0 --version
echo ""

# ================================================================
# 6. DeepStream 9.0
# ================================================================
# DeepStream — видеоаналитика NVIDIA.
# Устанавливается из .deb пакета (bare-metal, без Docker).
#
# СКАЧАТЬ ВРУЧНУЮ:
#   https://developer.nvidia.com/deepstream-getting-started
#   Файл: deepstream-9.0_9.0.0-1_amd64.deb

echo ""
echo "=== 6. DeepStream 9.0 ==="

DS_DEB="deepstream-9.0_9.0.0-1_amd64.deb"

if [ -f "$DS_DEB" ]; then
    sudo apt install -y ./$DS_DEB
    echo "[OK] DeepStream 9.0 установлен"
else
    echo ""
    echo "[!] Файл $DS_DEB не найден."
    echo "    Скачай вручную с developer.nvidia.com/deepstream-getting-started"
    echo "    Положи в текущую директорию и перезапусти этот блок."
    echo ""
    echo "    После скачивания:"
    echo "    sudo apt install -y ./$DS_DEB"
    echo ""
fi

# Проверка
echo ""
echo "Проверка DeepStream:"
deepstream-app --version 2>/dev/null || echo "DeepStream ещё не установлен — см. инструкцию выше"
echo ""

# ================================================================
# 7. Python 3.12 + виртуальное окружение
# ================================================================
# Python 3.12 — стабильный, совместим со всеми библиотеками.
# 3.13 тоже можно, но 3.12 надёжнее для PyTorch nightly.

echo ""
echo "=== 7. Python 3.12 ==="

sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Создать виртуальное окружение
VENV_DIR="$HOME/race_env"
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Обновить pip
pip install --upgrade pip setuptools wheel

# Прописать автоактивацию в bashrc
if ! grep -q "race_env" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Race Vision Python environment" >> ~/.bashrc
    echo "source $VENV_DIR/bin/activate" >> ~/.bashrc
fi

echo ""
echo "Проверка Python:"
python --version
which python
echo ""

# ================================================================
# 8. PyTorch nightly (поддержка sm_120 / Blackwell)
# ================================================================
# Stable PyTorch НЕ поддерживает RTX 5070 Ti (sm_120).
# Nightly builds содержат поддержку Blackwell.
#
# ВАЖНО: После установки — зафиксировать версию!

echo ""
echo "=== 8. PyTorch (nightly, cu12) ==="

pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu12

# Зафиксировать версию
pip freeze | grep -E "^torch" > "$HOME/torch_versions.txt"
echo "Версии PyTorch сохранены в ~/torch_versions.txt"
echo "Для восстановления: pip install -r ~/torch_versions.txt"

# Проверка
echo ""
echo "Проверка PyTorch + GPU:"
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPU:      {torch.cuda.get_device_name(0)}')
print(f'SM:       {torch.cuda.get_device_capability(0)}')
print(f'Доступен: {torch.cuda.is_available()}')

# Тест: создать тензор на GPU
x = torch.randn(1, 3, 800, 800, device='cuda')
print(f'Тензор на GPU: {x.shape} — OK')
"
echo ""

# ================================================================
# 9. Остальные Python-зависимости
# ================================================================

echo ""
echo "=== 9. Python-зависимости Race Vision ==="

pip install \
    ultralytics>=8.4.37 \
    onnx==1.21.0 \
    onnxruntime-gpu==1.24.4 \
    numpy==2.4.4 \
    opencv-python-headless>=4.12.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    websockets>=14.0 \
    posix-ipc>=1.1.1 \
    onnxsim \
    scipy \
    pillow \
    pyyaml \
    requests \
    aiofiles \
    python-multipart

# Проверка
echo ""
echo "Проверка зависимостей:"
python -c "
import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')
import onnx;        print(f'ONNX:        {onnx.__version__}')
import numpy;       print(f'NumPy:       {numpy.__version__}')
import cv2;         print(f'OpenCV:      {cv2.__version__}')
import fastapi;     print(f'FastAPI:     {fastapi.__version__}')
"

# TensorRT Python (в venv)
pip install tensorrt 2>/dev/null && \
    python -c "import tensorrt; print(f'TensorRT:    {tensorrt.__version__}')" || \
    echo "TensorRT Python — используется системный"

echo ""

# ================================================================
# 10. Клонировать проект и собрать C++ плагин
# ================================================================

echo ""
echo "=== 10. Сборка C++ плагина ==="

PROJECT_DIR="$HOME/race_kabirhan"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Клонируем проект..."
    git clone https://github.com/Mussylman/race_kabirhan.git "$PROJECT_DIR"
fi

cd "$PROJECT_DIR/deepstream"
mkdir -p build
cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8

make -j$(nproc)

echo ""
echo "Проверка сборки:"
ls -la lib*.so 2>/dev/null || echo "Плагин в текущей директории:"
ls -la *.so 2>/dev/null || echo "[!] .so файл не найден — проверь вывод cmake/make"
echo ""

# ================================================================
# 11. ПОЛНАЯ ПРОВЕРКА СТЕКА
# ================================================================

echo ""
echo "============================================"
echo "  ПОЛНАЯ ПРОВЕРКА СТЕКА"
echo "============================================"
echo ""

echo "--- Система ---"
echo "OS:     $(lsb_release -ds)"
echo "Kernel: $(uname -r)"
echo "CPU:    $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
echo "RAM:    $(free -h | awk '/Mem:/ {print $2}')"
echo ""

echo "--- NVIDIA ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: не найден"
echo ""

echo "--- CUDA ---"
nvcc --version 2>/dev/null | grep "release" || echo "nvcc: не найден"
echo ""

echo "--- cuDNN ---"
dpkg -l 2>/dev/null | grep cudnn | head -1 | awk '{print $2, $3}' || echo "cuDNN: не найден"
echo ""

echo "--- TensorRT ---"
dpkg -l 2>/dev/null | grep -i "tensorrt" | head -1 | awk '{print $2, $3}' || echo "TensorRT: не найден"
echo ""

echo "--- GStreamer ---"
gst-inspect-1.0 --version 2>/dev/null | head -1 || echo "GStreamer: не найден"
echo ""

echo "--- DeepStream ---"
deepstream-app --version 2>/dev/null || echo "DeepStream: не установлен"
echo ""

echo "--- Python ---"
python --version 2>/dev/null || echo "Python: не найден"
echo ""

echo "--- PyTorch ---"
python -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA:        {torch.version.cuda}')
print(f'GPU:         {torch.cuda.get_device_name(0)}')
print(f'SM:          {torch.cuda.get_device_capability(0)}')
print(f'Доступен:    {torch.cuda.is_available()}')
" 2>/dev/null || echo "PyTorch: не установлен"
echo ""

echo "--- Библиотеки ---"
python -c "
libs = {
    'ultralytics': 'ultralytics',
    'onnx': 'onnx',
    'onnxruntime': 'onnxruntime',
    'numpy': 'numpy',
    'opencv': 'cv2',
    'fastapi': 'fastapi',
    'PIL': 'PIL',
}
for name, mod in libs.items():
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', 'ok')
        print(f'{name:15s} {v}')
    except ImportError:
        print(f'{name:15s} НЕ УСТАНОВЛЕН')
" 2>/dev/null
echo ""

echo "--- C++ плагин ---"
PLUGIN="$PROJECT_DIR/deepstream/build/libnvdsinfer_yolov8_parser.so"
if [ -f "$PLUGIN" ]; then
    echo "Плагин: $PLUGIN ($(du -h $PLUGIN | cut -f1))"
else
    echo "Плагин: не собран"
fi
echo ""

echo "============================================"
echo "  УСТАНОВКА ЗАВЕРШЕНА"
echo "============================================"
echo ""
echo "Следующие шаги:"
echo "  1. Применить баг-фиксы Phase 1 (docs/PLAN_HYBRID_MIGRATION.md)"
echo "  2. Пересобрать TRT engine на новом GPU:"
echo "     cd $PROJECT_DIR"
echo "     python tools/export_trt.py"
echo "  3. Запустить тест на 1 камере:"
echo "     python api/server.py --cameras configs/cameras_1cam.json"
echo ""
