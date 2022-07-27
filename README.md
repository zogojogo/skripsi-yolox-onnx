# Indoor Object Detection using YOLOX-tiny on NVIDIA Jetson Series (ONNXRuntime Deployment)

To reproduce u can directly build the dockerfile especially if want to deploy to jetson devices. But if only want to try on local, just run the main.py directly with usual steps.

# Try on Local
```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 main.py -h
```

# Try on Jetson Series (via Docker)
## Build Docker
```bash
docker build -t zogojogo/yolox_skripsi_jetson:latest .
```

## How to Run Docker Image
```bash
sh runDocker.sh
```

## To run the main program
```bash
python3 main.py -h
```
