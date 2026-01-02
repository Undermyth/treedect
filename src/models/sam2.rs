use fast_image_resize as fr;
use ndarray::{Array2, Array3, ArrayView2, Axis, Dimension, Ix4, array, s};
use ndarray_stats::QuantileExt;
use ort::execution_providers::{
    CPUExecutionProvider, DirectMLExecutionProvider, ROCmExecutionProvider, WebGPUExecutionProvider,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::models::batcher;
use crate::panels::canvas;

pub struct SAM2Model {
    pub encoder_session: Session,
    pub decoder_session: Session,
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
        initialize: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if initialize {
            log::info!("Initializing ONNX Runtime with execution provider");
            ort::init()
                .with_execution_providers([CPUExecutionProvider::default()
                    .build()
                    .error_on_failure()])
                .commit()?;
        }
        log::info!("Loading encoder model from: {}", encoder_path);
        let encoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;
        log::info!("Loading decoder model from: {}", decoder_path);
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;
        log::info!("Successfully loaded both encoder and decoder models");
        Ok(Self {
            model_rel,
            encoder_session,
            decoder_session,
            mask_threshold: 0.0,
        })
    }

    pub fn forward(
        &mut self,
        batch_image: batcher::SAM2Batch,
    ) -> Result<SAM2Output, Box<dyn std::error::Error>> {
        // image encoding
        // let start_time = std::time::Instant::now();
        let batch_size = batch_image.image.shape()[0];
        let patch_size = batch_image.original_size;
        let input_tensor = TensorRef::from_array_view(&batch_image.image)?;
        let output = self
            .encoder_session
            .run(ort::inputs!["image" => input_tensor])?;
        let high_res_feats_0 = &output["high_res_feats_0"];
        let high_res_feats_1 = &output["high_res_feats_1"];
        let image_embed = &output["image_embed"];
        let image_embed = image_embed.try_extract_array::<f32>()?;
        let high_res_feats_0 = high_res_feats_0.try_extract_array::<f32>()?;
        let high_res_feats_1 = high_res_feats_1.try_extract_array::<f32>()?;
        // let end_time = start_time.elapsed();
        // log::info!("Encoding time taken: {:?}", end_time);

        // mask decoding
        // let start_time = std::time::Instant::now();
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
            let embedding = TensorRef::from_array_view(&embedding)?;

            let high_res_0 = high_res_feats_0.slice(s![i, .., .., ..]);
            let high_res_0 = high_res_0.to_shape(high_res_0_shape).unwrap();
            let high_res_0 = TensorRef::from_array_view(&high_res_0)?;

            let high_res_1 = high_res_feats_1.slice(s![i, .., .., ..]);
            let high_res_1 = high_res_1.to_shape(high_res_1_shape).unwrap();
            let high_res_1 = TensorRef::from_array_view(&high_res_1)?;

            let point_coords = sampling_coords.slice(s![i, ..]);
            let point_coords = point_coords.to_shape([1, 1, 2]).unwrap();
            let point_coords = TensorRef::from_array_view(&point_coords)?;

            let point_labels = array![[1 as f32]];
            let point_labels = TensorRef::from_array_view(&point_labels)?;

            let output = self.decoder_session.run(ort::inputs![
                "image_embed" => embedding,
                "high_res_feats_0" => high_res_0,
                "high_res_feats_1" => high_res_1,
                "point_coords" => point_coords,
                "point_labels" => point_labels,
            ])?;

            let mask = &output["masks"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix4>()?;
            // let scores = &output["iou_predictions"].try_extract_array::<f32>()?;
            // let max_index = scores.argmax()?.as_array_view()[1];

            let mask: ArrayView2<f32> = mask.slice(s![0, 0, .., ..]);

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
        // let end_time = start_time.elapsed();
        // log::info!("Decoding time taken: {:?}", end_time);

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
