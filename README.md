# yolox-onnx

## Build Docker
```bash
docker build -t zogojogo/yolox_skripsi_jetson:latest .
```

## How to Run Docker Image
```bash
xhost +local:docker 
docker run -it --rm --net=host --runtime nvidia zogojogo/yolox_skripsi_jetson:latest
xhost -local:docker
```
