use fast_image_resize as fr;
use ndarray::{Array2, Array3, Array4, ArrayView2, Axis, array, s};
use numpy::{IntoPyArray, PyArray, PyArray4, PyArrayMethods, ToPyArray};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::models::batcher;
use crate::panels::canvas;

pub struct SAM2Model {
    pub sam2_onnx: Py<PyAny>,
    pub model_rel: usize,
    pub mask_threshold: f32,
}

pub struct SAM2Output {
    pub mask_logits: Array3<f32>,
    pub coordinates: Array2<usize>,
    pub patch_size: usize,
}

impl SAM2Model {
    pub fn from_path(
        model_rel: usize,
        encoder_path: &str,
        decoder_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let sam2_onnx = Python::attach(|py| {
            // Add onnxinfer directory to Python path
            let sys_path = py.import("sys")?.getattr("path")?;
            sys_path.call_method1("append", ("onnxinfer",))?;

            // Import sam2onnx module and create SAM2ONNX instance
            let sam2onnx_module = py.import("sam2onnx")?;
            let sam2onnx_class = sam2onnx_module.getattr("SAM2ONNX")?;
            let sam2_onnx = sam2onnx_class.call1((encoder_path, decoder_path))?;

            log::info!("Successfully loaded Python SAM2ONNX model");
            Ok::<Py<PyAny>, Box<dyn std::error::Error>>(sam2_onnx.into())
        })?;

        Ok(Self {
            model_rel,
            sam2_onnx,
            mask_threshold: 0.0,
        })
    }

    pub fn forward(
        &mut self,
        batch_image: batcher::SAM2Batch,
    ) -> Result<SAM2Output, Box<dyn std::error::Error>> {
        let batch_size = batch_image.image.shape()[0];
        let patch_size = batch_image.original_size;

        // image encoding using Python SAM2ONNX
        let (image_embed, high_res_feats_0, high_res_feats_1) = Python::attach(|py| {
            let input_array = batch_image.image.into_pyarray(py);
            let sam2_onnx = self.sam2_onnx.as_ref();

            // Call encode method
            let encoded_dict = sam2_onnx.call_method1(py, c_str!("encode"), (input_array,))?;
            let encoded_dict = encoded_dict.cast_bound::<PyDict>(py).unwrap();

            // Extract features from Python dict and convert to ndarray
            let high_res_feats_0 = encoded_dict
                .get_item("high_res_feats_0")
                .unwrap()
                .unwrap()
                .cast_into::<PyArray4<f32>>()
                .unwrap();
            let high_res_feats_1 = encoded_dict
                .get_item("high_res_feats_1")
                .unwrap()
                .unwrap()
                .cast_into::<PyArray4<f32>>()
                .unwrap();
            let image_embed = encoded_dict
                .get_item("image_embedding")
                .unwrap()
                .unwrap()
                .cast_into::<PyArray4<f32>>()
                .unwrap();

            let high_res_feats_0_array = high_res_feats_0.to_owned_array();
            let high_res_feats_1_array = high_res_feats_1.to_owned_array();
            let image_embed_array = image_embed.to_owned_array();

            Ok::<(Array4<f32>, Array4<f32>, Array4<f32>), Box<dyn std::error::Error>>((
                image_embed_array,
                high_res_feats_0_array,
                high_res_feats_1_array,
            ))
        })?;

        // mask decoding using Python SAM2ONNX
        let sampling_coords =
            batch_image.sampling_coords / (patch_size as f32) * (self.model_rel as f32);
        let mut masks = Array3::<f32>::uninit([batch_size, patch_size, patch_size]);
        let embedding_shape = [
            1,
            image_embed.shape()[1],
            image_embed.shape()[2],
            image_embed.shape()[3],
        ];
        let high_res_0_shape = [
            1,
            high_res_feats_0.shape()[1],
            high_res_feats_0.shape()[2],
            high_res_feats_0.shape()[3],
        ];
        let high_res_1_shape = [
            1,
            high_res_feats_1.shape()[1],
            high_res_feats_1.shape()[2],
            high_res_feats_1.shape()[3],
        ];
        for i in 0..batch_size {
            let embedding = image_embed.slice(s![i, .., .., ..]);
            let embedding = embedding.to_shape(embedding_shape).unwrap();

            let high_res_0 = high_res_feats_0.slice(s![i, .., .., ..]);
            let high_res_0 = high_res_0.to_shape(high_res_0_shape).unwrap();

            let high_res_1 = high_res_feats_1.slice(s![i, .., .., ..]);
            let high_res_1 = high_res_1.to_shape(high_res_1_shape).unwrap();

            let point_coords = sampling_coords.slice(s![i, ..]);
            let point_coords = point_coords.to_shape([1, 1, 2]).unwrap();

            let point_labels = array![[1 as f32]];

            // Call Python decode method
            let decoded_mask = Python::attach(|py| {
                let embedding_py = embedding.to_pyarray(py);
                let high_res_0_py = high_res_0.to_pyarray(py);
                let high_res_1_py = high_res_1.to_pyarray(py);
                let point_coords_py = point_coords.to_pyarray(py);
                let point_labels_py = point_labels.to_pyarray(py);

                let sam2_onnx = self.sam2_onnx.as_ref();
                let decoded_dict = sam2_onnx.call_method1(
                    py,
                    c_str!("decode"),
                    (
                        embedding_py,
                        high_res_0_py,
                        high_res_1_py,
                        point_coords_py,
                        point_labels_py,
                    ),
                )?;

                let decoded_dict = decoded_dict.cast_bound::<PyDict>(py).unwrap();

                let masks_py = decoded_dict
                    .get_item("masks")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyArray4<f32>>()
                    .unwrap();
                let masks_array = masks_py.to_owned_array();

                Ok::<Array4<f32>, Box<dyn std::error::Error>>(masks_array)
            })?;

            let mask: ArrayView2<f32> = decoded_mask.slice(s![0, 0, .., ..]);

            let mask = mask.as_standard_layout().into_owned();
            let (mask_vec, _) = mask.into_raw_vec_and_offset();
            let mask_bytes: Vec<u8> = bytemuck::cast_slice(&mask_vec).to_vec();
            let mask_image =
                fr::images::Image::from_vec_u8(256, 256, mask_bytes, fr::PixelType::F32);
            let mut scaled_mask =
                fr::images::Image::new(patch_size as u32, patch_size as u32, fr::PixelType::F32);
            let mut resizer = fr::Resizer::new();
            let result = resizer.resize(mask_image.as_ref().unwrap(), &mut scaled_mask, None);
            if let Err(e) = result {
                log::error!("Error resizing mask: {:?}", e);
            }
            let scaled_mask = scaled_mask.into_vec();
            let scaled_mask: &[f32] = bytemuck::cast_slice(&scaled_mask);
            let scaled_mask = scaled_mask.to_vec();
            let scaled_mask =
                ArrayView2::from_shape((patch_size, patch_size), scaled_mask.as_slice())?;
            scaled_mask.view().assign_to(masks.slice_mut(s![i, .., ..]));
        }

        unsafe {
            Ok(SAM2Output {
                mask_logits: masks.assume_init().to_owned(),
                coordinates: batch_image.coordinates,
                patch_size: patch_size,
            })
        }
    }

    pub fn decode_mask_to_palette(&self, output: &SAM2Output, palette: &mut canvas::Palette) {
        for (i, (mask_logit, coordinate)) in output
            .mask_logits
            .axis_iter(Axis(0))
            .zip(output.coordinates.axis_iter(Axis(0)))
            .enumerate()
        {
            let mask = mask_logit.mapv(|x| {
                (x > self.mask_threshold) as usize * (i + palette.num_patches + 1) as usize
            });
            let coord_y = coordinate[0] as usize;
            let coord_x = coordinate[1] as usize;
            let patch_size = output.patch_size;

            let mut palette_slice = palette.map.slice_mut(s![
                coord_y..coord_y + patch_size,
                coord_x..coord_x + patch_size
            ]);

            for ((y, x), &mask_val) in mask.indexed_iter() {
                if mask_val != 0 {
                    palette_slice[[y, x]] = mask_val as usize;
                }
            }
        }
        palette.num_patches += output.mask_logits.shape()[0];
    }
}
