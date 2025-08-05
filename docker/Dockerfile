FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Install system dependencies
RUN apt update && apt upgrade -y \
  && apt install -y \
    libopenblas-dev libsvm-dev liblapack-dev libblas-dev \
    sudo \
  && rm -rf /var/lib/apt/lists/*

# Download libift.so and compile code to learn FLIM Model and Extract Features

# Define build arguments for user creation
ARG USER_NAME=developer
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create group and user with specified IDs
RUN groupadd --gid ${GROUP_ID} ${USER_NAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}

# Set up working directory
ARG WORKDIR=/workdir
RUN mkdir -p ${WORKDIR} \
    && chown -R ${USER_NAME}:${USER_NAME} ${WORKDIR}

# Add custom binary path to PATH
ENV PATH="/workdir/lib_flim_mca/bin:$PATH"

# Switch to non-root user
USER ${USER_NAME}
WORKDIR ${WORKDIR}

# Set default command
CMD ["/bin/bash"]