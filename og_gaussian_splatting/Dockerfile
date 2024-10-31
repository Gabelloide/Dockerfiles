FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

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

# Upgrade pip and install required Python packages
RUN python3.12 -m pip install --upgrade pip 

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

# Clone the repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# Create venv and install dependencies for Gaussian Splatting
RUN python3.12 -m venv /python_venv \
    && . /python_venv/bin/activate \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install opencv-python matplotlib ninja plyfile tqdm joblib setuptools wheel \
    && pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization \
    && pip install -e gaussian-splatting/submodules/simple-knn \
    && deactivate

# VENV
ENV VIRTUAL_ENV=/python_venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install COLMAP dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libmetis-dev \
    libflann-dev \
    libsqlite3-dev \
    libceres-dev

# Clone COLMAP repository
RUN git clone https://github.com/colmap/colmap.git

RUN cd /colmap \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;90" .. \
    && make -j8 \
    && make install

WORKDIR /gaussian-splatting




