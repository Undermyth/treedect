use ndarray::{Array3, ArrayView2, array, s};
use opencv::{core as cv, prelude::*};
use ort::execution_providers::WebGPUExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::models::batcher;

pub struct SAM2Model {
    pub encoder_session: Session,
    pub decoder_session: Session,
    pub model_rel: usize,
}

impl SAM2Model {
    pub fn from_path(
        model_rel: usize,
        encoder_path: &str,
        decoder_path: &str,
        initialize: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if initialize {
            log::info!("Initializing ONNX Runtime with WebGPU execution provider");
            ort::init()
                .with_execution_providers([WebGPUExecutionProvider::default()
                    .build()
                    .error_on_failure()])
                .commit()?;
        }
        log::info!("Loading encoder model from: {}", encoder_path);
        let encoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;
        log::info!("Loading decoder model from: {}", decoder_path);
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;
        log::info!("Successfully loaded both encoder and decoder models");
        Ok(Self {
            model_rel,
            encoder_session,
            decoder_session,
        })
    }

    pub fn forward(
        &mut self,
        batch_image: batcher::SAM2Batch,
    ) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        // image encoding
        let start_time = std::time::Instant::now();
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
        let end_time = start_time.elapsed();
        log::info!("Encoding time taken: {:?}", end_time);

        // mask decoding
        let start_time = std::time::Instant::now();
        let sampling_coords =
            batch_image.sampling_coords / (patch_size as f32) * (self.model_rel as f32);
        let mut masks = Array3::<f32>::uninit([batch_size, patch_size, patch_size]);
        log::info!("{:?}", image_embed.shape());
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

            let mask = &output["masks"].try_extract_array::<f32>()?;
            let mask = mask.slice(s![0, 0, .., ..]);

            // convert to opencv for resizing and convert back
            // .to_shape([self.model_rel, self.model_rel, 1])
            // .unwrap();
            // let shape: Vec<i32> = mask.shape().iter().map(|&sz| sz as i32).collect();
            // let (channels, shape) = shape.split_last().unwrap();
            let mask = mask.as_standard_layout();
            let mat = cv::Mat::from_slice(mask.as_slice().unwrap())?;
            let mut scaled_mat = cv::Mat::default();
            let _ = opencv::imgproc::resize(
                &mat,
                &mut scaled_mat,
                cv::Size::new(patch_size as i32, patch_size as i32),
                0.0,
                0.0,
                0,
            );
            let scaled_mask = ArrayView2::from_shape(
                (patch_size, patch_size),
                scaled_mat.data_typed::<f32>().unwrap(),
            )?;
            scaled_mask.view().assign_to(masks.slice_mut(s![i, .., ..]));
        }
        let end_time = start_time.elapsed();
        log::info!("Decoding time taken: {:?}", end_time);
        unsafe { Ok(masks.assume_init().to_owned()) }
    }
}
