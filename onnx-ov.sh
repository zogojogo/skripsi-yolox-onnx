MODEL=yolox_tiny
H=416
W=416
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir openvino/${MODEL}/${H}x${W}/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir openvino/${MODEL}/${H}x${W}/FP16
mkdir -p openvino/${MODEL}/${H}x${W}/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m openvino/${MODEL}/${H}x${W}/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o openvino/${MODEL}/${H}x${W}/myriad/${MODEL}_${H}x${W}.blob