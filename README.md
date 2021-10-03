![](graphical_abstract.png)

# dStripe: slice artefact correction in diffusion MRI

This repository contains code and model weights for the method described in the paper
"**dStripe: slice artefact correction in diffusion MRI via constrained neural network**" by
Maximilian Pietsch, Daan Christiaens, Joseph V Hajnal, J-Donald Tournier, published in 
Medical Image Analysis, 2021, 102255, https://doi.org/10.1016/j.media.2021.102255.

Abstract: MRI scanner and sequence imperfections and advances in reconstruction and imaging techniques to increase motion robustness can lead to inter-slice intensity variations in Echo Planar Imaging. Leveraging deep convolutional neural networks as universal image filters, we present a data-driven method for the correction of acquisition artefacts that manifest as inter-slice inconsistencies, regardless of their origin. This technique can be applied to motion- and dropout-artefacted data by embedding it in a reconstruction pipeline. The network is trained in the absence of ground-truth data on, and finally applied to, the reconstructed multi-shell high angular resolution diffusion imaging signal to produce a corrective slice intensity modulation field. This correction can be performed in either motion-corrected or scattered source-space. We focus on gaining control over the learned filter and the image data consistency via built-in spatial frequency and intensity constraints. The end product is a corrected image reconstructed from the original raw data, modulated by a multiplicative field that can be inspected and verified to match the expected features of the artefact. In-plane, the correction approximately preserves the contrast of the diffusion signal and throughout the image series, it reduces inter-slice inconsistencies within and across subjects without biasing the data. We apply our pipeline to enhance the super-resolution reconstruction of neonatal multi-shell high angular resolution data as acquired in the developing Human Connectome Project.

Keywords: diffusion MRI; image artefact removal; venetian blind artefact

This repository is set up as a module to [MRtrix3](https://www.mrtrix.org/) and uses pytorch and a number of other python packages (see [environment.yml](environment.yml)).

# Usage for anatomical-space inference

For ease of use, we recommend using Docker as outlined below. dStripe supports CPU-only and single and multi-GPU usage. In the usage examples, input and output data are located in `~/data` and mapped to `/data` inside the Docker container. This can be adjusted by modifying the command line option `--volume ~/data/:/data`.

```bash
IMAGE_NAME=dstripe
IMAGE_TAG=0.1
```

## build docker image 

```bash
[ -z "$UID" ] && UID=$(id -u)
[ -z "$GID" ] && GID=$(id -g)
docker image build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --file docker/Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  .
```

Make sure you have sufficient RAM! You might need to increase Docker's runtime memory ([macOS](https://docs.docker.com/docker-for-mac/#memory), [windows](https://docs.docker.com/docker-for-windows/#advanced)).

## show dwidestripe usage

```bash
docker container run \
  --rm \
  --volume ~/data/:/data \
  $IMAGE_NAME:$IMAGE_TAG \
  dwidestripe
```

## dwidestripe ~/data/dwi.mif on the CPU (relatively slow)

```bash
docker container run --rm --volume ~/data/:/data \
  $IMAGE_NAME:$IMAGE_TAG \
  dwidestripe /data/dwi.mif /data/mask.mif /data/dstripe_field.mif -device cpu
```

## dwidestripe ~/data/dwi.mif on the GPU

For [GPU support](https://docs.docker.com/config/containers/resource_constraints/) add `--gpus` and replace `-device cpu` with for instance `-device 0,1` for using the first two CUDA-capable GPUs.

```bash
docker container run --rm --volume ~/data/:/data --gpus \
  $IMAGE_NAME:$IMAGE_TAG \
  dwidestripe /data/dwi.mif /data/mask.mif /data/dstripe_field.mif -device 0,1
```

## apply dStripe field

```bash
docker container run --rm --volume ~/data/:/data \
  $IMAGE_NAME:$IMAGE_TAG \
  mrcalc /data/dwi.mif /data/dstripe_field.mif -mult /data/dwi_destriped.mif
```

## acknowledgement

```bibtex
@article{pietsch2021dStripe,
title = {dStripe: slice artefact correction in diffusion MRI via constrained neural network},
journal = {Medical Image Analysis},
pages = {102255},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102255},
author = {Maximilian Pietsch and Daan Christiaens and Joseph V Hajnal and J-Donald Tournier}
}
```
