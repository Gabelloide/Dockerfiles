FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04

# Set non-interactive frontend for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl wget \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake \
    ffmpeg x264 x265 imagemagick \
    git aria2 \
    nano \
    libgl1-mesa-glx libglib2.0-0 \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# CUDA 
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"


# # Create venv and install dependencies for Gaussian Splatting
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.11 && \
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH
RUN conda init

# # Clone the repository
# COPY . /triangle-splatting
RUN git clone https://github.com/trianglesplatting/triangle-splatting --recursive
WORKDIR /triangle-splatting

RUN conda env create -f requirements.yaml

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate triangle_splatting \
    && bash compile.sh \
    && cd submodules/simple-knn && pip install .
