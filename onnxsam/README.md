## SAM2 & DINOv2 ONNX Conversion

This is a subproject which converts SAM2 & DINOv2 models into ONNX format. `treedect` utilizes ONNX models for portable and efficient inference, and also for better integration with Rust ecosystem. Since the conversion process is not trivial, this subproject is maintained separately. All the converted models are are designed to access through web, which are uploaded to [Hugging Face](https://huggingface.co/Activator/vision_onnx).

ONNX exportation is very sensitive to environment, so it's recommended to have a dedicated environment for this purpose. As verified, we should have python <= 3.11 (for compatibility with `onnxsim` binaries), and torch == 2.5.1.

**All the export code requires invasive modification of the model**, as stated below.

### SAM2 ONNX Exportation
The export script of SAM2 is borrowed from [samexporter](https://github.com/vietanhdev/samexporter). Exporting SAM2 requires installing the SAM2 repo as editable. It is recommended to clone the SAM2 repo and use `pip install -e .` for installation. To control the torch version, you can remove the version constraint in `pyproject.toml` and `setup.py`. Also we have to sightly modify the code in SAM2 for ONNX exportation on CPU to work properly on Windows. All the modifications are in `sam2.patch`. You can apply the patch with `git apply ../sam2.patch` in the root directory of SAM2.

The example ONNX exportation command is as follows:
```python
python .\export_sam2.py --checkpoint facebook/sam2-hiera-small --output_encoder output_models/sam2_hiera_small.encoder.onnx --output_decoder output_models/sam2_hiera_small.decoder.onnx --model_type sam2_hiera_small --opset 18
```

### DINOv2 ONNX Exportation
The modification for DINOv2 comes from [this issue](https://github.com/facebookresearch/dinov2/issues/19), mainly for ONNX compatibility with interpolation operators. You will need to handle this by yourself, as the code for model will be under `~/.cache/torch/hub`. We can enable torch dynamo to avoid the problem but that is not friendly to `onnxsim`.