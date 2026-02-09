import onnx
import onnxruntime as ort
import numpy as np
import torch
import time

def test_dinov2_onnx_model(model_path="output_models/dinov2_vitb_reg.onnx"):
    """
    测试DINOv2 ONNX模型的推理功能
    """
    # 加载ONNX模型
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully loaded ONNX model from {model_path}")

    # 创建ONNX Runtime会话，设置执行提供者为WebGPU
    ort_session = ort.InferenceSession(model_path, providers=['DmlExecutionProvider'])
    print("ONNX Runtime session created successfully with DML execution provider")

    # 准备测试输入数据 (batch_size=1, channels=3, height=448, width=448)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"Input name: {input_name}, Input shape: {input_shape}")

    # 创建随机输入数据，与export_dinov2.py中的输入保持一致
    np.random.seed(0)
    dummy_input = np.random.randn(2, 3, 448, 448).astype(np.float32)
    print(f"Input tensor shape: {dummy_input.shape}")

    # 运行推理
    start_time = time.time()
    outputs = ort_session.run(None, {input_name: dummy_input})
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    output = outputs[0]  # 获取输出
    print(f"Output tensor shape: {output.shape}")
    print(f"Output tensor type: {output.dtype}")
    print("Inference completed successfully!")

    # 验证输出
    print(f"Output stats - Min: {output.min()}, Max: {output.max()}, Mean: {output.mean()}")
    # print(output)

    return output

if __name__ == "__main__":
    print("Testing DINOv2 ONNX model...")
    output = test_dinov2_onnx_model()
