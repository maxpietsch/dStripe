FROM nvidia/cuda:11.1-base-ubuntu18.04
# FROM mrtrix3/mrtrix3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bzip2 \
        clang \
        curl \
        ca-certificates \
        git \
        libeigen3-dev \
        libfftw3-dev \
        libgl1-mesa-dev \
        libpng-dev \
        libqt5opengl5-dev \
        libqt5svg5-dev \
        libtiff5-dev \
        qt5-default \
        sudo \
        vim \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Make bash the default shell
SHELL [ "/bin/bash", "--login", "-c" ]


# Create a non-root user
ARG username=me
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

COPY environment.yml /tmp/
RUN chown $UID:$GID /tmp/environment.yml

COPY docker/entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

USER $USER

# install miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CONDA_DIR $HOME/miniconda3
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh  && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash

# install mrtrix3
ARG MRTRIX3_GIT_COMMITISH="3.0.1"
ARG MRTRIX3_CONFIGURE_FLAGS="-nogui"
ARG MRTRIX3_BUILD_FLAGS=""
# -persistent -nopaginate
ARG MAKE_JOBS="12"
WORKDIR $HOME/mrtrix3
RUN git clone -b $MRTRIX3_GIT_COMMITISH --depth 1 https://github.com/MRtrix3/mrtrix3.git . \
    && ./configure $MRTRIX3_CONFIGURE_FLAGS \
    && NUMBER_OF_PROCESSORS=$MAKE_JOBS ./build $MRTRIX3_BUILD_FLAGS bin/mrconvert bin/mrinfo bin/mrcat bin/mrcalc \
    && rm -rf tmp

# create project dir
ENV PROJECT_DIR $HOME/dStripe
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

COPY --chown=$UID:$GID ../. $PROJECT_DIR/
RUN ln -s $HOME/mrtrix3/build $PROJECT_DIR/build
RUN ln -s $HOME/mrtrix3/bin/mrtrix3.py $PROJECT_DIR/bin/mrtrix3.py

ENV PATH=$PROJECT_DIR/bin/:$PATH

# build the conda environment
ENV ENV_PREFIX $HOME/env
RUN mkdir $ENV_PREFIX
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

CMD ["/bin/bash"]