
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ARG username

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /main
RUN mkdir /main/home
WORKDIR /main

# RUN git clone https://github.com/gerritgr/nextaid.git

# Create a non-root user and switch to it
#RUN adduser --disabled-password --gecos '' --shell /bin/bash $username \
#    && chown -R $username:$username /main
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER $username

# All users can use /home/user as their home directory
ENV HOME=/main/home
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

# Set up the Conda environment (using Miniforge)
ENV PATH=$HOME/mambaforge/bin:$PATH
COPY environment.yml /main/environment.yml
RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh \
    && chmod +x ~/mambaforge.sh \
    && ~/mambaforge.sh -b -p ~/mambaforge \
    && rm ~/mambaforge.sh \
    && mamba env update -n base -f /main/environment.yml \
    #&& rm /main/environment.yml \
    && mamba clean -ya
    
# install jax
#RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # should be don in env file
