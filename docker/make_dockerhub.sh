#!/bin/bash
# docker rmi $(docker images -qa -f 'dangling=true')
# docker system prune
set -x

IMAGE_NAME=maxpietsch/dstripe
IMAGE_TAG=0.1

cd ../
docker image build \
  --file docker/Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  .


docker push $IMAGE_NAME:$IMAGE_TAG

