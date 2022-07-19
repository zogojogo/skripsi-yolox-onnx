import skripsi
import os
import argparse

model = skripsi.InferOnnx(model_path='./model/yoloxv6.onnx')

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--path', type=str, help='Input Path')

def prepare_input(path):
    os.system('rm -r ./mAP-skripsi/input/detection-results/*')
    os.system('rm -r ./mAP-skripsi/input/ground-truth/*')
    os.system('rm -r ./mAP-skripsi/input/images-optional/*')
    os.system(f'cp -r {path}/*.xml ./mAP-skripsi/input/ground-truth/')
    os.system(f'cp -r {path}/*.jpg ./mAP-skripsi/input/images-optional/')
    os.system('python ./mAP-skripsi/scripts/extra/convert_gt_xml.py')

def run_batches(path):
    os.system('rm -r ./outputs/*')
    model.run_batches(path)
    os.system('cp -r ./outputs/* ./mAP-skripsi/input/detection-results/')
    os.system('python ./mAP-skripsi/scripts/extra/intersect-gt-and-dr.py')

if __name__ == "__main__":
    os.system('git clone https://github.com/zogojogo/mAP-skripsi.git')
    args = parser.parse_args()
    benchmark_path = args.path
    prepare_input(benchmark_path)
    run_batches(benchmark_path)
    os.system('python ./mAP-skripsi/main.py -na')