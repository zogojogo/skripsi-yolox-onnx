import skripsi
import argparse
import os

model = skripsi.InferOnnx(model_path='model/yoloxv6.onnx')

parser = argparse.ArgumentParser(description='Infer ONNX model')
parser.add_argument('--mode', type=str, help='Toggle Run Image/Video/Batch')
parser.add_argument('--path', type=str, help='Input Path')
parser.add_argument('--vis', help='Toggle Visualize', default=False, dest='vis', action='store_true')
parser.add_argument('--txt', help='Toggle Write Box to .txt', default=False, dest='txt', action='store_true')

args = parser.parse_args()
if args.mode == 'video':
    model.run_video(args.path, enable_vis=True)
elif args.mode == 'image':
    model.run(args.path, enable_vis=args.vis, write_output=args.txt)
elif args.mode == 'batches':
    os.system('rm -r ./outputs/*')
    model.run_batches(args.path)