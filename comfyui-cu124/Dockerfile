################################################################################
# Dockerfile that builds 'yanwk/comfyui-boot:cu124-megapak'
# A big all-in-one package with many custom nodes.
# Using CUDA 12.4, Python 3.12, GCC 13.
# The container will be running in root (easy for rootless deploy).
################################################################################

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

LABEL maintainer="YAN Wenkun <code@yanwk.fun>"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    gcc g++ make cmake \
    ffmpeg x264 x265 \
    git aria2 \
    nano \
    fonts-noto-cjk fonts-noto-color-emoji \
    libgl1-mesa-glx libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py \
    && rm -rf /var/lib/apt/lists/*


# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip and install required Python packages
RUN python3.12 -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages \
        --upgrade pip wheel setuptools Cython numpy

# PyTorch, xFormers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip list \
    && pip install \
        --upgrade pip wheel setuptools \
    && pip install \
        xformers torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple

# Dependencies for frequently-used
# (Do this firstly so PIP won't be solving too many deps at one time)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages \
        -r https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt \
        -r https://raw.githubusercontent.com/crystian/ComfyUI-Crystools/main/requirements.txt \
        -r https://raw.githubusercontent.com/cubiq/ComfyUI_essentials/main/requirements.txt \
        -r https://raw.githubusercontent.com/Fannovel16/comfyui_controlnet_aux/main/requirements.txt \
        -r https://raw.githubusercontent.com/jags111/efficiency-nodes-comfyui/main/requirements.txt \
        -r https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Pack/Main/requirements.txt \
        -r https://raw.githubusercontent.com/ltdrdata/ComfyUI-Impact-Subpack/main/requirements.txt \
        -r https://raw.githubusercontent.com/ltdrdata/ComfyUI-Inspire-Pack/main/requirements.txt \
        -r https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/requirements.txt

################################################################################
# GCC 13 
# Required for compiling CUDA 12.4-related code.

#RUN apt-get update && \
#    add-apt-repository ppa:ubuntu-toolchain-r/test && \
#    apt-get update && \
#    apt-get install -y gcc-13 g++-13 && \
#    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 90 && \
#    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 90 && \
#    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 90 && \
#    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 90

################################################################################
#  Dependencies for more, with few hand-pick:
# 'cupy-cuda12x' for Frame Interpolation
# 'compel lark' for smZNodes
# 'torchdiffeq' for DepthFM
# 'fairscale' for APISR
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages \
        -r https://raw.githubusercontent.com/cubiq/ComfyUI_FaceAnalysis/main/requirements.txt \
        -r https://raw.githubusercontent.com/cubiq/ComfyUI_InstantID/main/requirements.txt \
        -r https://raw.githubusercontent.com/Fannovel16/ComfyUI-Frame-Interpolation/main/requirements-no-cupy.txt \
        cupy-cuda12x \
        -r https://raw.githubusercontent.com/FizzleDorf/ComfyUI_FizzNodes/main/requirements.txt \
        -r https://raw.githubusercontent.com/kijai/ComfyUI-KJNodes/main/requirements.txt \
        -r https://raw.githubusercontent.com/melMass/comfy_mtb/main/requirements.txt \
        -r https://raw.githubusercontent.com/MrForExample/ComfyUI-3D-Pack/main/requirements.txt \
        -r https://raw.githubusercontent.com/storyicon/comfyui_segment_anything/main/requirements.txt \
        -r https://raw.githubusercontent.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/main/requirements.txt \
        compel lark torchdiffeq fairscale \
        python-ffmpeg

# 1. Fix ONNX Runtime "missing CUDA provider". Also add support for CUDA 12.1.
#    Ref: https://onnxruntime.ai/docs/install/
# 2. Fix MediaPipe's broken dep (protobuf<4).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages \
        --force-reinstall onnxruntime-gpu \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
        --extra-index-url https://pypi.org/simple \
    && pip install --break-system-packages \
        mediapipe

# Bind libs (.so files)
# Even we have CUDA installed by Zypper, we still need to install CUDA libs for Python in order to run PyTorch.
# What's more, NVIDIA's openSUSE15 repo didn't provide CuDNN & NCCL, we have to use Python packages anyway.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}\
:/usr/lib64/python3.12/site-packages/torch/lib\
:/usr/lib/python3.12/site-packages/nvidia/cuda_cupti/lib\
:/usr/lib/python3.12/site-packages/nvidia/cuda_runtime/lib\
:/usr/lib/python3.12/site-packages/nvidia/cudnn/lib\
:/usr/lib/python3.12/site-packages/nvidia/cufft/lib\
:/usr/lib/python3.12/site-packages/nvidia/cublas/lib\
:/usr/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib\
:/usr/lib/python3.12/site-packages/nvidia/curand/lib\
:/usr/lib/python3.12/site-packages/nvidia/cusolver/lib\
:/usr/lib/python3.12/site-packages/nvidia/cusparse/lib\
:/usr/lib/python3.12/site-packages/nvidia/nccl/lib\
:/usr/lib/python3.12/site-packages/nvidia/nvjitlink/lib\
:/usr/lib/python3.12/site-packages/nvidia/nvtx/lib"

COPY builder-scripts/.  /builder-scripts/

# 1. Deps for ComfyUI & custom nodes
# 2. Fix ONNX Runtime "missing CUDA provider". Also add support for CUDA 12. Ref: https://onnxruntime.ai/docs/install/
# 3. Fix MediaPipe's broken dep (protobuf<4).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        -r /builder-scripts/pak3.txt \
    && pip install \
        -r /builder-scripts/pak5.txt \
    && pip install \
        -r /builder-scripts/pak7.txt \
    && pip uninstall --break-system-packages --yes \
        onnxruntime-gpu \
    && pip --no-cache-dir install --break-system-packages \
        onnxruntime-gpu \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
        --extra-index-url https://pypi.org/simple \
    && pip install \
        mediapipe \
    && pip list

################################################################################

COPY runner-scripts/.  /runner-scripts/
RUN chmod +x /runner-scripts/entrypoint.sh \
    && chmod +x /runner-scripts/install_comfy.sh

USER root
VOLUME /root
EXPOSE 40188
ENV CLI_ARGS=""

RUN ./runner-scripts/install_comfy.sh
RUN cd /root/ComfyUI && git checkout v0.3.7

CMD ["bash","/runner-scripts/entrypoint.sh"]
