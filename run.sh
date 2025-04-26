# docker build . -t object-detector-ssd-yolo8

xhost +local:

# Then run the container
docker run -it --rm \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY=:0 \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -v ${PWD}:/app \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw \
    --network=host \
    --gpus all \
    object-detector-ssd-yolo8
