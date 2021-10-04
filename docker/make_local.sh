#!/bin/bash

[ -z "$UID" ] && UID=$(id -u)
[ -z "$GID" ] && GID=$(id -g)
OSTYPE=$(uname)

docker image build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --build-arg ostype=$OSTYPE \
  --file docker/Dockerfile_user \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  .
