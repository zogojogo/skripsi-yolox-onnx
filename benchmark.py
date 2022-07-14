import skripsi
import os

model = skripsi.InferOnnx(model_path='./model/yoloxv5_1.onnx')

def prepare_input(path):
    os.system('rm -r ./mAP/input/detection-results/*')
    os.system('rm -r ./mAP/input/ground-truth/*')
    os.system('rm -r ./mAP/input/images-optional/*')
    os.system(f'cp -r {path}/*.xml ./mAP/input/ground-truth/')
    os.system(f'cp -r {path}/*.jpg ./mAP/input/images-optional/')
    os.system('python ./mAP/scripts/extra/convert_gt_xml.py')

def run_batches(path):
    os.system('rm -r ./outputs/*')
    model.run_batches(path)
    os.system('cp -r ./outputs/* ./mAP/input/detection-results/')
    os.system('python ./mAP/scripts/extra/intersect-gt-and-dr.py')

if __name__ == "__main__":
    # os.system('git clone https://github.com/Cartucho/mAP.git')
    benchmark_path = './Benchmark_Data/TestingKost/test'
    prepare_input(benchmark_path)
    run_batches(benchmark_path)
    os.system('python ./mAP/main.py -na')