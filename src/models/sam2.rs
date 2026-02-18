use std::sync::{Arc, Mutex};
use std::vec;

use fast_image_resize as fr;
use fast_image_resize::images::Image;
use image::RgbImage;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, Ix2, Ix4, array, s};
use ndarray_stats::QuantileExt;
use ort::ep::{CPU, DirectML};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::panels::palette;

pub struct SAM2Model {
    pub encoder_session: Session,
    pub decoder_session: Session,
    pub model_rel: usize,
    /// image embedding and size saved at run time
    pub image_embed: Option<Array4<f32>>,
    pub high_res_feats_0: Option<Array4<f32>>,
    pub high_res_feats_1: Option<Array4<f32>>,
    pub image_size: Option<usize>,
}

pub struct SAM2Output {
    pub mask_logits: Array3<f32>,
    pub coordinates: Array2<usize>,
    pub patch_size: usize,
}

pub struct SAM2Batch {
    // [B, C, H, W] where H = W = 1024
    pub image: Array4<f32>,
    /// [B, 2], left up corner [x, y] of the patches.
    /// The coordinate is in the format [width, height], aligned with the image coordinate system.
    /// The whole patch will be [x..x+patch_size, y..y+patch_size]
    pub coordinates: Array2<usize>,
    /// [B, 2], the coordinate of sampling points in the image.
    /// ***The coordinate is in the format [height, width] to align with DL models.***
    /// The sampling points are recorded before scaling. All scaling will be processed
    /// within SAM2 model.
    pub sampling_coords: Array2<f32>,
    /// original size of patch
    pub original_size: usize,
}

impl SAM2Batch {
    #[allow(dead_code)]
    pub fn print(&self) {
        println!("Image shape: {:?}", self.image.shape());
        println!("Coordinates:\n{}", self.coordinates);
        println!("Sampling coords:\n{}", self.sampling_coords);
        println!("Original size: {}", self.original_size);
    }
}

pub struct SAM2Batcher {
    idx: usize,
    batch_size: usize,
    patch_size: usize,
    model_rel: usize, // resolution of SAM2 model, default to 1024
    sampling_points: Vec<[usize; 2]>,
    raw_image: Arc<Mutex<RgbImage>>,
    mean: Array1<f32>,
    std: Array1<f32>,
}

impl SAM2Batcher {
    pub fn new(
        batch_size: usize,
        patch_size: usize,
        sampling_points: Vec<[usize; 2]>,
        raw_image: Arc<Mutex<RgbImage>>,
    ) -> Self {
        Self {
            idx: 0,
            batch_size,
            patch_size,
            model_rel: 1024,
            sampling_points,
            raw_image,
            mean: Array1::from_vec(vec![0.485, 0.456, 0.406]),
            std: Array1::from_vec(vec![0.229, 0.224, 0.225]),
        }
    }
}

impl Iterator for SAM2Batcher {
    type Item = SAM2Batch;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.sampling_points.len() {
            return None;
        }
        let start_idx = self.idx;
        let end_idx = (self.idx + self.batch_size).min(self.sampling_points.len());
        let actual_batch_size = end_idx - start_idx;
        let mut batch =
            Array4::<f32>::uninit((actual_batch_size, self.model_rel, self.model_rel, 3));
        let mut sampling_coords = Array2::<f32>::uninit((actual_batch_size, 2));
        let mut coordinates = Array2::<usize>::uninit((actual_batch_size, 2));

        // Get lock on the image for the duration of this batch iteration
        let raw_image = self.raw_image.lock().unwrap();
        let image_width = raw_image.width() as usize;
        let image_height = raw_image.height() as usize;

        for i in start_idx..end_idx {
            let [x, y] = self.sampling_points[i];
            let mut start_x = x.saturating_sub(self.patch_size / 2);
            let mut start_y = y.saturating_sub(self.patch_size / 2);
            let end_x = (start_x + self.patch_size).min(image_width);
            let end_y = (start_y + self.patch_size).min(image_height);
            if end_x == image_width {
                start_x = end_x - self.patch_size;
            }
            if end_y == image_height {
                start_y = end_y - self.patch_size;
            }
            array![start_y, start_x]
                .view()
                .assign_to(coordinates.slice_mut(s![i - start_idx, ..]));
            array![(x - start_x) as f32, (y - start_y) as f32]
                .view()
                .assign_to(sampling_coords.slice_mut(s![i - start_idx, ..]));
            let patch = raw_image
                .view(
                    start_x as u32,
                    start_y as u32,
                    (end_x - start_x) as u32,
                    (end_y - start_y) as u32,
                )
                .to_image();
            let patch = DynamicImage::ImageRgb8(patch);
            let mut dst_image = Image::new(
                self.model_rel as u32,
                self.model_rel as u32,
                fr::PixelType::U8x3,
            );
            let mut resizer = fr::Resizer::new();
            let _ = resizer.resize(&patch, &mut dst_image, None);
            let shape = (self.model_rel, self.model_rel, 3);
            let mut sample = Array3::from_shape_vec(
                shape,
                dst_image.into_vec().into_iter().map(|b| b as f32).collect(),
            )
            .unwrap();
            sample = (sample / 255.0 - self.mean.to_shape([1, 1, 3]).unwrap())
                / self.std.to_shape([1, 1, 3]).unwrap();
            sample
                .view()
                .assign_to(batch.slice_mut(s![i - start_idx, .., .., ..]));
        }
        self.idx += actual_batch_size;
        // log::info!("Proceeding batch to {}", self.idx);
        unsafe {
            let batch = SAM2Batch {
                image: batch
                    .assume_init()
                    .permuted_axes([0, 3, 1, 2])
                    .as_standard_layout()
                    .to_owned(),
                coordinates: coordinates.assume_init().to_owned(),
                sampling_coords: sampling_coords.assume_init().to_owned(),
                original_size: self.patch_size,
            };
            // batch.print();
            return Some(batch);
        }
    }
}

pub fn squeeze_3d_to_4d(array: Array3<f32>) -> Array4<f32> {
    let shape = [1, array.shape()[0], array.shape()[1], array.shape()[2]];
    let array = array.to_shape(shape).unwrap();
    return array.to_owned();
}

impl SAM2Model {
    pub fn from_path(
        model_rel: usize,
        encoder_path: &str,
        decoder_path: &str,
        is_cpu: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Loading encoder model from: {}", encoder_path);
        let encoder_session = Session::builder()?
            .with_execution_providers([if is_cpu {
                CPU::default().build()
            } else {
                DirectML::default().build().error_on_failure()
            }])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;
        log::info!("Loading decoder model from: {}", decoder_path);
        let decoder_session = Session::builder()?
            .with_execution_providers([if is_cpu {
                CPU::default().build()
            } else {
                DirectML::default().build().error_on_failure()
            }])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;
        log::info!("Successfully loaded both encoder and decoder models");
        log::info!("Intrinsic resolution of the model: {}", model_rel);
        Ok(Self {
            model_rel,
            encoder_session,
            decoder_session,
            image_embed: None,
            high_res_feats_0: None,
            high_res_feats_1: None,
            image_size: None,
        })
    }

    pub fn preprocess(&self, image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Array4<f32> {
        let image = DynamicImage::ImageRgb8(image);
        let mut dst_image = Image::new(
            self.model_rel as u32,
            self.model_rel as u32,
            fr::PixelType::U8x3,
        );
        let mut resizer = fr::Resizer::new();
        let _ = resizer.resize(&image, &mut dst_image, None);
        let shape = (self.model_rel, self.model_rel, 3);
        let sample = Array3::from_shape_vec(
            shape,
            dst_image.into_vec().into_iter().map(|b| b as f32).collect(),
        )
        .unwrap();
        let sample = sample / 255.0 - array![[[0.485, 0.456, 0.406]]];
        let sample = sample / array![[[0.229, 0.224, 0.225]]];
        return squeeze_3d_to_4d(sample)
            .permuted_axes([0, 3, 1, 2])
            .as_standard_layout()
            .to_owned();
    }

    pub fn forward(
        &mut self,
        batch_image: SAM2Batch,
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
            let scores = &output["iou_predictions"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()?;
            let (_, max_index) = scores.argmax()?;
            let mut max_index = if scores[(0, 0)] > scores[(0, 1)] {
                0
            } else {
                1
            };
            // let decoded_mask = mask.mapv(|x| (x > 0.0) as usize);
            // let valid_area = decoded_mask.sum();
            // if max_index == 1 {
            //     if (valid_area as f32) > 0.8 * self.model_rel as f32 * self.model_rel as f32 {
            //         max_index = 0;
            //     }
            //     // if (valid_area as f32) < 0.1 * self.model_rel as f32 * self.model_rel as f32 {
            //     //     max_index = 2;
            //     // }
            // }
            // if max_index == 0
            //     && (valid_area as f32) < 0.1 * self.model_rel as f32 * self.model_rel as f32
            // {
            //     max_index = 1;
            // }

            let mask: ArrayView2<f32> = mask.slice(s![0, max_index, .., ..]);

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

    pub fn set_image(
        &mut self,
        image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let size = image.width() as usize;
        let image_tensor = self.preprocess(image);
        let input_tensor = TensorRef::from_array_view(&image_tensor)?;
        let output = self
            .encoder_session
            .run(ort::inputs!["image" => input_tensor])?;
        let image_embed = &output["image_embed"];
        let high_res_feats_0 = &output["high_res_feats_0"];
        let high_res_feats_1 = &output["high_res_feats_1"];
        let image_embed = image_embed
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned();
        let high_res_feats_0 = high_res_feats_0
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned();
        let high_res_feats_1 = high_res_feats_1
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned();
        self.image_embed = Some(image_embed);
        self.high_res_feats_0 = Some(high_res_feats_0);
        self.high_res_feats_1 = Some(high_res_feats_1);
        self.image_size = Some(size);
        Ok(())
    }

    pub fn decode(
        &mut self,
        coords: [usize; 2], // [width, height]
    ) -> Result<Array2<usize>, Box<dyn std::error::Error>> {
        assert!(
            !self.image_embed.is_none(),
            "Image embeddings not set. Call `set_image` first."
        );
        assert!(
            !self.high_res_feats_0.is_none(),
            "High resolution features 0 not set. Call `set_image` first."
        );
        assert!(
            !self.high_res_feats_1.is_none(),
            "High resolution features 1 not set. Call `set_image` first."
        );
        assert!(
            !self.image_size.is_none(),
            "Image size not set. Call `set_image` first."
        );
        let image_size = self.image_size.unwrap();
        let embedding = TensorRef::from_array_view(self.image_embed.as_ref().unwrap())?;
        let high_res_0 = TensorRef::from_array_view(self.high_res_feats_0.as_ref().unwrap())?;
        let high_res_1 = TensorRef::from_array_view(self.high_res_feats_1.as_ref().unwrap())?;
        let point_coords = array![[[coords[0] as f32, coords[1] as f32]]];
        let point_coords = point_coords / (image_size as f32) * (self.model_rel as f32);
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
        let scores = &output["iou_predictions"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()?;
        let (_, mut max_index) = scores.argmax()?;
        let selected_mask: ArrayView2<f32> = mask.slice(s![0, max_index, .., ..]);
        let decoded_mask = selected_mask.mapv(|x| (x > 0.0) as usize);
        let valid_areas = decoded_mask.sum();
        if max_index >= 1
            && (valid_areas as f32) > 0.5 * (self.model_rel as f32) * (self.model_rel as f32)
        {
            max_index = 0;
        }
        max_index = 0;
        let mask: ArrayView2<f32> = mask.slice(s![0, max_index, .., ..]);
        let mask = mask.as_standard_layout().into_owned();
        let (mask_vec, _) = mask.into_raw_vec_and_offset();
        let mask_bytes: Vec<u8> = bytemuck::cast_slice(&mask_vec).to_vec();
        let mask_image = fr::images::Image::from_vec_u8(256, 256, mask_bytes, fr::PixelType::F32);
        let mut scaled_mask =
            fr::images::Image::new(image_size as u32, image_size as u32, fr::PixelType::F32);
        let mut resizer = fr::Resizer::new();
        let _ = resizer.resize(mask_image.as_ref().unwrap(), &mut scaled_mask, None);
        let scaled_mask = scaled_mask.into_vec();
        let scaled_mask: &[f32] = bytemuck::cast_slice(&scaled_mask);
        let scaled_mask = scaled_mask.to_vec();
        let scaled_mask = ArrayView2::from_shape((image_size, image_size), scaled_mask.as_slice())?;
        Ok(scaled_mask.mapv(|x| (x > 0.0) as usize))
    }

    pub fn tiled_diffuse_merge_scan(
        &mut self,
        raw_image: Arc<Mutex<RgbImage>>,
        patch_size: usize,
        lumin_filt: u8,
        x_scan_interval: usize,
        y_scan_interval: usize,
        merge_thr: f32,
    ) -> palette::Palette {
        let image_size = raw_image.lock().unwrap().width() as usize;
        let stride = patch_size / 2;
        let x_stride = x_scan_interval;
        let y_stride = y_scan_interval;
        let merge_thr = merge_thr;
        let corrupt_thr = 0.8;
        let black_thr: u8 = lumin_filt;
        let total_area = (patch_size * patch_size) as f32;

        let mut palette = palette::Palette::new(image_size);

        for start_x in (0..(image_size - stride)).step_by(stride) {
            // height direction
            for start_y in (0..(image_size - stride)).step_by(stride) {
                // width direction
                let outer_loop_start = std::time::Instant::now();
                log::info!("patch position: {start_x}, {start_y}");
                let patch_start_x = if start_x + patch_size > image_size {
                    image_size - patch_size
                } else {
                    start_x
                };
                let patch_start_y = if start_y + patch_size > image_size {
                    image_size - patch_size
                } else {
                    start_y
                };

                let patch = raw_image
                    .lock()
                    .unwrap()
                    .view(
                        patch_start_y as u32,
                        patch_start_x as u32,
                        patch_size as u32,
                        patch_size as u32,
                    )
                    .to_image();
                let set_image_start = std::time::Instant::now();
                let result = self.set_image(patch);
                log::info!("Set image time: {:?}", set_image_start.elapsed());
                if let Err(e) = result {
                    log::error!("Error: {e}");
                }

                let sample_start_y = if patch_start_y == 0 { 0 } else { stride };
                let sample_start_x = if patch_start_x == 0 { 0 } else { stride };

                let mut total_samples = 0u64;
                let mut total_decode_time = std::time::Duration::ZERO;
                let mut total_detect_overlap_time = std::time::Duration::ZERO;
                let mut total_modify_segment_time = std::time::Duration::ZERO;

                for sample_x in (sample_start_x..patch_size).step_by(y_stride) {
                    // height direction
                    for sample_y in (sample_start_y..patch_size).step_by(x_stride) {
                        // width direction
                        let image = raw_image.lock().unwrap();
                        let pixel = image.get_pixel(sample_y as u32, sample_x as u32);
                        if is_black(pixel[0], pixel[1], pixel[2], black_thr) {
                            continue;
                        }
                        drop(image);

                        total_samples += 1;
                        let decode_start = std::time::Instant::now();
                        let decode_result = self.decode([sample_y, sample_x]);
                        total_decode_time += decode_start.elapsed();

                        if let Err(e) = &decode_result {
                            log::error!("Error: {e}");
                        }
                        let mask = decode_result.unwrap();
                        let area = mask.sum() as f32;
                        if area / total_area > corrupt_thr {
                            continue;
                        }

                        let detect_start = std::time::Instant::now();
                        let detect_result = palette.detect_overlap(
                            &mask,
                            [patch_start_x, patch_start_y],
                            patch_size,
                        );
                        total_detect_overlap_time += detect_start.elapsed();

                        let modify_start = std::time::Instant::now();
                        match detect_result.iter().max_by_key(|&(_, v)| v) {
                            Some((&segment_id, &overlap_area)) => {
                                let old_area = palette.get_area(segment_id) as f32;
                                if (overlap_area as f32) / old_area > merge_thr
                                    || (overlap_area as f32) / area > merge_thr
                                {
                                    // // anti disturb
                                    // if area > 1.5 * old_area {
                                    //     let shifted_sample_x = sample_x.saturating_sub(5);
                                    //     // let shifted_sample_y = sample_y.saturating_sub(5);
                                    //     let disturb_result =
                                    //         self.decode([sample_y, shifted_sample_x]);
                                    //     let disturb_mask = disturb_result.unwrap();
                                    //     let overlap = (disturb_mask * &mask).sum() as f32;
                                    //     if overlap < 0.8 * total_area {
                                    //         continue;
                                    //     }
                                    // }
                                    palette.expand_segment(
                                        segment_id,
                                        mask,
                                        [patch_start_x, patch_start_y],
                                        patch_size,
                                    );
                                } else {
                                    palette.add_segment(
                                        mask,
                                        [patch_start_x, patch_start_y],
                                        patch_size,
                                    );
                                }
                            }
                            None => {
                                palette.add_segment(
                                    mask,
                                    [patch_start_x, patch_start_y],
                                    patch_size,
                                );
                            }
                        }
                        total_modify_segment_time += modify_start.elapsed();
                    }
                }

                let total_elapsed = outer_loop_start.elapsed();
                if total_samples > 0 {
                    let avg_decode = total_decode_time.div_f64(total_samples as f64);
                    let avg_detect = total_detect_overlap_time.div_f64(total_samples as f64);
                    let avg_modify = total_modify_segment_time.div_f64(total_samples as f64);
                    log::info!(
                        "Timing [{start_x}, {start_y}]: samples={}, average={:?}, decode_avg={:?}, detect_overlap_avg={:?}, modify_segment_avg={:?}",
                        total_samples,
                        total_elapsed,
                        avg_decode,
                        avg_detect,
                        avg_modify
                    );
                }
            }
        }
        palette
    }
}

fn is_black(r: u8, g: u8, b: u8, thr: u8) -> bool {
    let luminance = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
    (r < thr && g < thr && b < thr) || (luminance < (thr as f32))
}
