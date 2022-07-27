FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime

#Install all packages
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-tk

#Python Dependencies
RUN pip3 install matplotlib opencv_contrib_python tqdm gdown

# Set working directory
WORKDIR /home/app

#Main Codes
COPY benchmark.py main.py runDocker.sh requirements.txt /home/app/

#Download Benchmark Data
RUN gdown 1hgO202fmC-cNs3kOSJltm3tKx-UexkjQ
RUN unzip benchmark.zip

#Inference Codes
COPY skripsi/ /home/app/skripsi/

#Model & Benchmark Files
COPY testing/ /home/app/testing/
COPY model/yoloxv6.onnx /home/app/model/
RUN mkdir outputs
RUN mkdir cache

#Install ONNXRuntime
RUN wget https://nvidia.box.com/shared/static/2sv2fv1wseihaw8ym0d4srz41dzljwxh.whl -O onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl && \
pip3 install onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl

CMD ["/bin/bash"]
