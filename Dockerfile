FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
# We add the 'deadsnakes' PPA so we can still install older Python 3.8 on Ubuntu 22.04
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    git curl wget build-essential ninja-build \
    python3.8 python3.8-dev python3.8-distutils \
    libgl1-mesa-glx libglib2.0-0 \
    sudo x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.8 /usr/bin/pip
# 2. Handle User Access
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} devuser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash devuser && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER devuser
WORKDIR /home/devuser/TRASE
ENV PATH="/home/devuser/.local/bin:${PATH}"

# 3. Pre-install heavy base dependencies
RUN pip3.8 install --user torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install Python packages required by TRASE
RUN pip3.8 install --user opencv-python plyfile tqdm scipy scikit-learn lpips \
    imageio[ffmpeg] dearpygui kmeans_pytorch hdbscan scikit-image bitarray

# 4. Copy local repo and compile CUDA submodules permanently
COPY --chown=devuser:devuser . /home/devuser/TRASE

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Force PyTorch to use the specific CUDA version for compilation
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN pip install --user --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"

RUN pip3.8 install --user ./submodules/diff-gaussian-rasterization && \
    pip3.8 install --user ./submodules/simple-knn

# Compile Grounded-Segment-Anything (Assuming it is in the root directory)
RUN cd dependency/Grounded-Segment-Anything && \
    export AM_I_DOCKER=True && \
    export BUILD_WITH_CUDA=True && \
    pip3.8 install --user ./segment_anything && \
    pip3.8 install --user ./GroundingDINO

RUN pip3.8 install --user supervision==0.21.0

RUN cd ..

CMD ["/bin/bash"]