import onnxruntime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import glob
import tqdm

from .utils import multiclass_nms, demo_postprocess, ResizeWithAspectRatio
from .visualize import box_to_txt, vis
from .classes import CLASSES

class InferOnnx():
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = self.init_model()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def init_model(self):
        sess = onnxruntime.InferenceSession(self.model_path, providers=[('TensorrtExecutionProvider', {'trt_engine_cache_enable':True, 'trt_engine_cache_path':"/home/app/cache"})][('TensorrtExecutionProvider', {'trt_engine_cache_enable':True, 'trt_engine_cache_path':"/home/app/cache"})])
        return sess

    def preproc(self, input, input_size, swap=(2, 0, 1)):
        if len(input.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        img = np.array(input)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        image = padded_img

        image = image.astype(np.float32)
        image = image[:, :, ::-1]
        image /= 255.0
        image -= self.mean
        image /= self.std
        image = image.transpose(swap)

        image = np.ascontiguousarray(image, dtype=np.float32)
        return image, r

    def predict_onnx(self, img_path):
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        img, ratio = self.preproc(img_path, (416, 416))
        start = time.time()
        output = self.sess.run(None, {input_name: img[None, :, :, :]})
        end = time.time()
        inference_time = end - start
        pred = demo_postprocess(output[0], (416, 416))[0]
        return pred, ratio, inference_time

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

    def visualize(self, dets, origin_img, conf_thr=0.3):
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            result_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=conf_thr, class_names=CLASSES)
            return result_img
        else:
            return origin_img

    def run(self, img_path, enable_vis=False, write_output = False):
        origin_img = cv2.imread(img_path)
        predictions, ratio, time = self.predict_onnx(origin_img)
        dets = self.postprocess(predictions, ratio)
        if enable_vis:
            result = self.visualize(dets, origin_img)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            plt.imshow(result)
            plt.show()
            print('Inference Time : {:.3f} ms'.format(time * 1000))
            print('FPS : {:.2f}'.format(1 /(time)))
            cv2.imwrite('./outputs/prediction.jpg', result)
        if write_output:
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                box_to_txt(img_path, final_boxes, final_scores, final_cls_inds, CLASSES, conf=0.5)
        return dets

    def run_video(self, video_path, enable_vis=False):
        cap = cv2.VideoCapture(video_path)
        imageWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        imageHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./outputs/output.mp4', fourcc, fps, (imageWidth, imageHeight))
        while True:
            ret, frame = cap.read()
            if ret:
                predictions, ratio, time = self.predict_onnx(frame)
                print('Inference time: {:.2f} ms'.format(time))
                print('FPS : {:.2f}'.format(1 / (time)))
                dets = self.postprocess(predictions, ratio)
                if enable_vis:
                    result = self.visualize(dets, frame, conf_thr=0.5)
                    resize = ResizeWithAspectRatio(result, width=900, height=900)
                    cv2.imshow('result', resize)
                    out.write(result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def run_batches(self, dir_path):
        for img_path in tqdm.tqdm(glob.glob(dir_path + '/*.jpg')):
            # print(img_path)
            self.run(img_path, enable_vis=False, write_output=True)
