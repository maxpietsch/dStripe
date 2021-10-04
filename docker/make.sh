#!/bin/bash
# docker rmi $(docker images -qa -f 'dangling=true')
# docker system prune
set -x

IMAGE_NAME=dstripe
IMAGE_TAG=0.1

[ -z "$UID" ] && UID=$(id -u)
[ -z "$GID" ] && GID=$(id -g)
OSTYPE=$(uname)

cd ../
docker image build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --build-arg ostype=$OSTYPE \
  --file docker/Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  .
