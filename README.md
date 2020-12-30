# dStripe

```
IMAGE_NAME=dstripe
IMAGE_TAG=0.1
```

## build docker image (make sure you have sufficient RAM )

```
cd docker
[ -z "$UID" ] && UID=$(id -u)
[ -z "$GID" ] && GID=$(id -g)
docker image build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --file Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  ../
```

You might need to increase Docker's runtime memory ([macOS](https://docs.docker.com/docker-for-mac/#memory), [windows](https://docs.docker.com/docker-for-windows/#advanced)).

## show dwidestripe usage

```
docker container run \
  --rm \
  --volume ~/data/:/data \
  $IMAGE_NAME:$IMAGE_TAG \
  dwidestripe
```

## dwidestripe ~/data/dwi.mif on the CPU (slow)

```
docker container run \
  --rm \
  --volume ~/data/:/data \
  $IMAGE_NAME:$IMAGE_TAG \
  dwidestripe /data/dwi.mif /data/mask.mif /data/dstripe_field.mif -device cpu
```

## dwidestripe ~/data/dwi.mif on the GPU

For [GPU support](https://docs.docker.com/config/containers/resource_constraints/) add `--gpus` and replace `-device cpu` with `-device 0,1` for CUDA-capable GPUs 0 and 1. 

```
docker container run \
  --rm \
  --gpus \
  --volume ~/data/:/data \
  $IMAGE_NAME:$IMAGE_TAG \
  dwidestripe /data/dwi.mif /data/mask.mif /data/dstripe_field.mif -device 0,1
```

## apply dStripe field

```
docker container run \
  --rm \
  --volume ~/data/:/data \
  mrcalc /data/dwi.mif /data/dstripe_field.mif -mult /data/dwi_destriped.mif
```
