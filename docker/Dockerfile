FROM mrtrix3/mrtrix3@sha256:0de10cfddc4b5a99dc99e33cdf838c29f1cc73cf61622fafa9b355ea698493cd AS mrtrix301

FROM nvidia/cuda:11.1-base-ubuntu18.04

LABEL MAINTAINER="Max Pietsch <maximilian.pietsch@kcl.ac.uk>"

# fix "mesg: ttyname failed: Inappropriate ioctl for device" warning
RUN cp /etc/profile /root/.profile

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bzip2 \
        curl \
        ca-certificates \
        git \
        libfftw3-dev \
        # libgl1-mesa-dev \
        libpng-dev \
        # libqt5opengl5-dev \
        # libqt5svg5-dev \
        libtiff5-dev \
        # qt5-default \
        # sudo \
        # vim \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Make bash the default shell
# SHELL [ "/bin/bash", "--login", "-c" ]

COPY environment.yml /tmp/

COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

USER $USER

# install miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CONDA_DIR /opt/miniconda3
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh  && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init --system bash

COPY --from=mrtrix301 /opt/mrtrix3 /opt/mrtrix3

# create project dir
ENV PROJECT_DIR /opt/dStripe
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

COPY . $PROJECT_DIR/
RUN ln -s /opt/mrtrix3/build $PROJECT_DIR/build
RUN ln -s /opt/mrtrix3/bin/mrtrix3.py $PROJECT_DIR/bin/mrtrix3.py

ENV PATH=$PROJECT_DIR/bin/:$PATH

# build the conda environment
ENV ENV_PREFIX /opt/env
RUN mkdir $ENV_PREFIX
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

WORKDIR /
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

CMD ["/bin/bash"]
