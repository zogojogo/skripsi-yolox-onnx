MODEL=yolox_tiny
H=416
W=416
openvino2tensorflow \
--model_path openvino/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
# --output_edgetpu \
--weight_replacement_config replace_tiny.json
mv saved_model saved_model_${MODEL}_${H}x${W}