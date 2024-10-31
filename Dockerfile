# Use the official NVIDIA CUDA base image
#FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the timezone to Singapore
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install tzdata without interactive prompts
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y tzdata

# Install dependencies including wget
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    wget \
    libcublas-12-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-key del 7fa2af80

# Install the correct CUDA keyring
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

# Explicitly add the NVIDIA repository with the Signed-By option
RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" | tee /etc/apt/sources.list.d/cuda.list

# Update repository and install cuDNN
RUN apt-get update \
    && apt-get -y install libcudnn8 libcudnn8-dev

# Install TensorFlow with GPU support
RUN pip3 install --upgrade pip && \
    pip3 install tensorflow datasets nltk tensorflow scipy gensim

# Set the environment variable for XLA to find libdevice
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/nvvm/libdevice


# Set default command
# CMD ["python3", "/tf/code/script.py"]
CMD ["/bin/bash"]