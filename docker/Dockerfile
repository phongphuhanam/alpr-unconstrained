FROM nvcr.io/nvidia/tensorflow:19.08-py2

RUN DEBIAN_FRONTEND=noninteractive

RUN python -m pip install --upgrade \
  pip \
  setuptools


RUN apt-get update && apt-get install -y --no-install-recommends \
	sudo rsync  imagemagick \
    ffmpeg \
    libfreetype6-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install  --use-feature=2020-resolver \
    opencv-python==4.2.0.32 keras==2.3.0

ENV USERNAME msis_dev
ENV HOME /home/$USERNAME
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        mkdir -p /etc/sudoers.d && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Replace 1000 with your user/group id
        usermod  --uid 1000 $USERNAME && \
        groupmod --gid 1000 $USERNAME

RUN ARCH="$(dpkg --print-architecture)" && \
    curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.5/fixuid-0.5-linux-$ARCH.tar.gz" | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: msis_dev\ngroup: msis_dev\n" > /etc/fixuid/config.yml


# install SDK Manager
USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENTRYPOINT ["fixuid", "bash"]

