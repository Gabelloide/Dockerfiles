FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set non-interactive frontend for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl wget \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    gcc g++ make cmake \
    ffmpeg x264 x265 imagemagick \
    git aria2 \
    nano \
    libgl1-mesa-glx libglib2.0-0 \
    libavcodec-extra \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# CUDA
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Fix for libs (.so files)
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}\
:/usr/lib64/python3.12/site-packages/torch/lib\
:/usr/lib/python3.12/site-packages/nvidia/cuda_cupti/lib\
:/usr/lib/python3.12/site-packages/nvidia/cuda_runtime/lib\
:/usr/lib/python3.12/site-packages/nvidia/cudnn/lib\
:/usr/lib/python3.12/site-packages/nvidia/cufft/lib"

# Repo
RUN git clone --recursive https://github.com/graphdeco-inria/on-the-fly-nvs.git

# Create venv and install dependencies for Gaussian Splatting
RUN python3.12 -m venv /python_venv \
    && . /python_venv/bin/activate \
    && pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 \
    && pip install cupy-cuda12x \
    && pip install opencv-python matplotlib ninja plyfile tqdm joblib setuptools wheel lpips \
    && pip install -e on-the-fly-nvs/submodules/diff-gaussian-rasterization \
    && mkdir -p on-the-fly-nvs/submodules/simple-knn/simple_knn \
    && touch on-the-fly-nvs/submodules/simple-knn/simple_knn/__init__.py \
    && pip install -e on-the-fly-nvs/submodules/simple-knn \
    && pip install -e on-the-fly-nvs/submodules/fused-ssim \
    && pip install -e on-the-fly-nvs/submodules/graphdecoviewer \
    && deactivate
