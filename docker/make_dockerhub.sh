#!/bin/bash
# docker rmi $(docker images -qa -f 'dangling=true')
# docker system prune
set -x
set -e

IMAGE_NAME=maxpietsch/dstripe


cd ../

v=$(grep MRTRIX_PROJECT_VERSION src/project_version.h)
IMAGE_TAG=${v#*MRTRIX_PROJECT_VERSION }
IMAGE_TAG=${IMAGE_TAG//\"/}

docker image build \
  --file docker/Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  .


docker push $IMAGE_NAME:$IMAGE_TAG

