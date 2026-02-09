import torch
import torch.nn as nn
import onnx
import onnxsim
import numpy as np

class DINOv2Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        B, nc, w, h = x.shape
        x = self.model.patch_embed(x)
        x = torch.cat((self.model.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.model.interpolate_pos_encoding(x, w, h)
        x = torch.cat(
            (
                x[:, :1],
                self.model.register_tokens.expand(B, -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )
        for blk in self.model.blocks:
            x = blk(x)
        x_norm = self.model.norm(x)

        return x_norm[:, self.model.num_register_tokens + 1 :]



np.random.seed(0)
dummy_input = np.random.randn(2, 3, 448, 448).astype(np.float32)
x = torch.from_numpy(dummy_input)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
print(model)
model = DINOv2Model(model)
model(x)
# batch_dim = torch.export.Dim("batch", min=1, max=1024)
# dynamic_shapes = {"x": {0: batch_dim}}
filename = "output_models/dinov2_vitb_reg.onnx"
torch.onnx.export(
    model,
    x,
    filename,
    dynamo=False,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["img"],
    output_names=["patch_tokens"],
    dynamic_axes={"img": {0: "batch"}},
    # external_data=False
    # dynamic_shapes=dynamic_shapes
)
onnx_model = onnx.load(filename)
model_simp, check = onnxsim.simplify(onnx_model)
assert check
onnx.save(model_simp, filename)
