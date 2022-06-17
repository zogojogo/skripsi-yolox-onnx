#%%
import onnxruntime
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt

import time

from utils import multiclass_nms, demo_postprocess, ResizeWithAspectRatio
from visualize import vis
from classes import CLASSES
#%%
class InferOnnx():
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = self.init_model()

    def preprocess_img(self, img):
        transform = A.Compose([
            A.Resize(height=416, width=416, interpolation=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        r = min(416 / img.shape[0], 416 / img.shape[1])
        img = transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, r

    def init_model(self):
        sess = onnxruntime.InferenceSession(self.model_path)
        return sess

    def predict_onnx(self, model_path, img_path):
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        img, ratio = self.preprocess_img(img_path)
        output = self.sess.run([output_name], {input_name: img})
        pred = demo_postprocess(output[0], (416, 416))[0]
        return pred, ratio

    def postprocess(self, predictions, ratio, nms_thr=0.5):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=0.1)
        return dets

    def visualize(self, dets, origin_img, conf_thr=0.5):
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            result_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=conf_thr, class_names=CLASSES)
            return result_img
        else:
            return origin_img

    def run(self, img_path, enable_vis=False):
        origin_img = cv2.imread(img_path)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        predictions, ratio = self.predict_onnx(model_path, origin_img)
        dets = self.postprocess(predictions, ratio)
        if enable_vis:
            result = self.visualize(dets, origin_img)
            plt.imshow(result)
            plt.show()
        return dets

    def run_video(self, video_path, enable_vis=False):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start = time.time()
                predictions, ratio = self.predict_onnx(model_path, frame)
                end = time.time()
                print('Inference time: {}'.format(end - start))
                print('FPS : {}'.format(1/(end - start)))
                dets = self.postprocess(predictions, ratio)
                if enable_vis:
                    result = self.visualize(dets, frame)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    resize = ResizeWithAspectRatio(result, width=1200)
                    cv2.imshow('result', resize)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        

#%%
if __name__ == '__main__':
    model_path = './yolox_v2.onnx'
    model = InferOnnx(model_path)
    model.run('./office_2.jpg', enable_vis=True)
    # model.run_video('./test_kos.mp4', enable_vis=True)
# %%
