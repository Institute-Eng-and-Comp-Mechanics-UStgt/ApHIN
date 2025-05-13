# Docker configuration for ApHIN - Autoencoder-based port-Hamiltonian Identification Networks
# 
# For end users:
# ==============
# Build image:
# $ docker build -t aphin .
# 
# Run container without GUI support:
# $ docker run -it aphin
#
# Run container with GUI support (assuming that the host OS is Linux):
# $ xhost +local:root
# $ docker run -it --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" aphin
# $ xhost -local:root
#
# See <https://wiki.ros.org/docker/Tutorials/GUI> for alternative solutions.
# Similar solutions are available for Windows or macOS as host OS.
#
# Alternatively, start the container using docker compose:
# $ xhost +local:root
# $ docker compose run aphin
# $ xhost -local:root
#
# Terminate the container with `exit`
#
# For developers:
# ===============
# Push image to registry (after docker login):
# $ docker push <insert-organization-here>/aphin
#####################################

# Set the base image
# This container uses Tensorflow 2.18.0 without GPU support - Feel free to change to your environment
FROM tensorflow/tensorflow:2.18.0

# Define variables
ENV WORKSPACE_DIR="/home"
ENV PROJECT_DIR="${WORKSPACE_DIR}/aphin"
ENV GITHUB_ORG="Institute-Eng-and-Comp-Mechanics-UStgt"
ENV GITHUB_REPO="ApHIN"
ENV GITHUB_BRANCH="main"

# Install dependencies
RUN apt update && \
    apt install -y git python3-tk dvipng texlive-latex-extra texlive-fonts-recommended cm-super qt6-base-dev libxcb-cursor0

# Clone demonstrator from GitHub and install dependencies
ADD https://api.github.com/repos/${GITHUB_ORG}/${GITHUB_REPO}/git/refs/heads/main version.json
RUN cd ${WORKSPACE_DIR} && \
    git clone -b ${GITHUB_BRANCH} "https://github.com/${GITHUB_ORG}/${GITHUB_REPO}.git" ${PROJECT_DIR} && \
    cd ${PROJECT_DIR} && \
    pip install -e .

# Set working directory
WORKDIR ${PROJECT_DIR}

# Start bash
CMD ["/bin/bash"]
