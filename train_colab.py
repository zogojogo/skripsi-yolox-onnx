!git clone https://github.com/roboflow-ai/YOLOX.git
%cd YOLOX
!pip3 install -U pip && pip3 install -r requirements.txt
!pip3 install -v -e .  
!pip uninstall -y torch torchvision torchaudio
# # May need to change in the future if Colab no longer uses CUDA 11.0
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
  
%cd /content/
!git clone https://github.com/NVIDIA/apex
%cd apex
!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

%cd /content/
!pip install roboflow

dataset = project.version(2).download("voc")

%cd YOLOX
!python3 voc_txt.py "/content/YOLOX/datasets/VOCdevkit/"
!python3 voc_txt.py "/content/YOLOX/datasets/VOCdevkitt/"
%mkdir "/content/YOLOX/datasets/VOCdevkit/VOC2012"
!cp -r "/content/YOLOX/datasets/VOCdevkit/VOC2007/." "/content/YOLOX/datasets/VOCdevkit/VOC2012"

%%writetemplate /content/YOLOX/yolox/data/datasets/voc_classes.py

VOC_CLASSES = (
  'bed', 'chair', 'door', 'elevator', 'fire-extinguisher', 'person', 'sofa', 'stair', 'table', 'television', 'trashbin', 'window'
)

class Exp(MyExp):
  def __init__(self):
    super(Exp, self).__init__()
    self.num_classes = 12
    self.max_epochs = 300
    self.depth = 0.33
    self.width = 0.375
    self.scale = (0.5, 1.5)
    self.random_size = (10, 20)
    self.input_size = (416,416)
    self.test_size = (416,416)
    self.enable_mixup = False
    
%cd /content/
!wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny.pth
%cd /content/YOLOX/

!python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 24 --fp16 -o -c './yolox_tiny.pth'

!python3 tools/export_onnx.py --output-name yolox_tiny.onnx -n yolox-tiny -c ./YOLOX_Outputs/yolox_tiny.pth
