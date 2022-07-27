# yolox-onnx

## Build Docker
```bash
docker build -t zogojogo/yolox_skripsi_jetson:latest .
```

## How to Run Docker Image
```bash
xhost +local:docker 
XSOCK=/tmp/.X11-unix 
XAUTH=/tmp/.docker.xauth 
touch $XAUTH 
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" --volume="$XSOCK:$XSOCK:rw" zogojogo/yolox_skripsi_jetson:1.0.1
xhost -local:docker
```
