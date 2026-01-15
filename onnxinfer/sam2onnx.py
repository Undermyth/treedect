import onnxruntime as ort
from typing import Dict
import numpy as np

class SAM2ONNX:
    def __init__(self, encoder_path: str, decoder_path: str):
        self.encoder_session = ort.InferenceSession(
            encoder_path, providers=["DmlExecutionProvider"]
        )
        model_outputs = self.encoder_session.get_outputs()
        self.encoder_output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]
        self.decoder_session = ort.InferenceSession(
            decoder_path, providers=["DmlExecutionProvider"]
        )
        model_outputs = self.decoder_session.get_outputs()
        self.decoder_output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]
    def encode(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        output = self.encoder_session.run(
            self.encoder_output_names, {"image": image}
        )
        return {
            "high_res_feats_0": output[0],
            "high_res_feats_1": output[1],
            "image_embedding": output[2],
        }
    def decode(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        output = self.decoder_session.run(
            self.decoder_output_names,
            {
                "image_embed": image_embed,
                "high_res_feats_0": high_res_feats_0,
                "high_res_feats_1": high_res_feats_1,
                "point_coords": point_coords,
                "point_labels": point_labels,
            },
        )
        return {
            "masks": output[0],
            "scores": output[1],
        }

if __name__ == '__main__':
    encoder_path = r"C:\Users\Activator\code\treedect\output_models\sam2_hiera_small.encoder.onnx"
    decoder_path = r"C:\Users\Activator\code\treedect\output_models\sam2_hiera_small.decoder.onnx"
    sam2onnx = SAM2ONNX(encoder_path, decoder_path)
    image = np.ones((4, 3, 1024, 1024)).astype(np.float32)
    result = sam2onnx.encode(image)
    print(result)
