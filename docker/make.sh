IMAGE_NAME=dstripe
IMAGE_TAG=0.1
set -x
docker image build \
  --build-arg username=$USER \
  --build-arg uid=$(id -u) \
  --build-arg gid=$(id -g) \
  --file Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  ../



# cd ~/dStripe
# IMAGE_NAME=dstripe
# IMAGE_TAG=0.1
# docker container run \
#   --rm \
#   --volume /tmp:/home/$USER/data \
#   $IMAGE_NAME:$IMAGE_TAG \
#   dwidestripe
