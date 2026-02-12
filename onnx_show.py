import os
import onnx

# 遍历output_models目录下以sam3开头的所有onnx文件
output_models_dir = "output_models"
for filename in os.listdir(output_models_dir):
    if filename.startswith("sam2.1") and filename.endswith(".onnx"):
        onnx_path = os.path.join(output_models_dir, filename)
        print(f"\n{'='*60}")
        print(f"ONNX文件: {filename}")
        print('='*60)

        # 加载ONNX模型
        model = onnx.load(onnx_path)

        # 打印模型信息
        print(f"\n模型图名称: {model.graph.name}")
        print(f"Opset版本: {model.opset_import}")

        # 打印输入信息
        print(f"\n输入信息:")
        for i, input_tensor in enumerate(model.graph.input):
            elem_type = input_tensor.type.tensor_type.elem_type
            shape = [dim.dim_value if dim.dim_value != 0 else -1
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  输入 {i}: {input_tensor.name}")
            print(f"    数据类型: {elem_type}")
            print(f"    形状: {shape}")

        # 打印输出信息
        print(f"\n输出信息:")
        for i, output_tensor in enumerate(model.graph.output):
            elem_type = output_tensor.type.tensor_type.elem_type
            shape = [dim.dim_value if dim.dim_value != 0 else -1
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  输出 {i}: {output_tensor.name}")
            print(f"    数据类型: {elem_type}")
            print(f"    形状: {shape}")

        # 打印节点数量
        print(f"\n模型节点数量: {len(model.graph.node)}")
        print(f"初始化器数量: {len(model.graph.initializer)}")
