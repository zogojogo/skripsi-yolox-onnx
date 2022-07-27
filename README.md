# yolox-onnx

## Build Docker
```bash
docker build -t zogojogo/yolox_skripsi_jetson:latest .
```

## How to Run Docker Image
```bash
xhost +local:docker 
docker run -it --rm --net=host --runtime nvidia zogojogo/yolox_skripsi_jetson:1.0.1
xhost -local:docker
```
