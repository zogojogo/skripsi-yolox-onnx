FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime

# Install all packages
RUN apt-get update -y && apt-get upgrade -y && apt-get dist-upgrade &&\
    apt-get install ffmpeg libsm6 libxext6 libxrender1 -y 
    apt-get install -y \
    python3-dev libpython3-dev python3-tk liblapack-dev build-essential 

#Python Dependencies
RUN pip install --upgrade pip
RUN pip3 install matplotlib opencv_contrib_python tqdm

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
COPY benchmark/ /home/app/benchmark/
COPY testing/ /home/app/testing/
COPY model/yoloxv6.onnx /home/app/model/
COPY outputs/ /home/app/outputs/

# Install ONNXRuntime
RUN wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl && \
pip3 install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl

CMD ["/bin/bash"]
