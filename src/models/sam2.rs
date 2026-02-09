use std::sync::{Arc, Mutex};

use fast_image_resize as fr;
use fast_image_resize::images::Image;
use image::RgbImage;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, Axis, Ix2, Ix4, array, s};
use ndarray_stats::QuantileExt;
use ort::ep::DirectML;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::panels::palette;

pub struct SAM2Model {
    pub encoder_session: Session,
    pub decoder_session: Session,
    pub model_rel: usize,
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

impl SAM2Model {
    pub fn from_path(
        model_rel: usize,
        encoder_path: &str,
        decoder_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Loading encoder model from: {}", encoder_path);
        let encoder_session = Session::builder()?
            .with_execution_providers([DirectML::default().build().error_on_failure()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;
        log::info!("Loading decoder model from: {}", decoder_path);
        let decoder_session = Session::builder()?
            .with_execution_providers([DirectML::default().build().error_on_failure()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;
        log::info!("Successfully loaded both encoder and decoder models");
        log::info!("Intrinsic resolution of the model: {}", model_rel);
        Ok(Self {
            model_rel,
            encoder_session,
            decoder_session,
        })
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
            // let (_, max_index) = scores.argmax()?;
            let mut max_index = if scores[(0, 0)] > scores[(0, 1)] {
                0
            } else {
                1
            };
            let decoded_mask = mask.mapv(|x| (x > 0.0) as usize);
            let valid_area = decoded_mask.sum();
            if max_index == 1 {
                if (valid_area as f32) > 0.8 * self.model_rel as f32 * self.model_rel as f32 {
                    max_index = 0;
                }
                // if (valid_area as f32) < 0.1 * self.model_rel as f32 * self.model_rel as f32 {
                //     max_index = 2;
                // }
            }
            if max_index == 0
                && (valid_area as f32) < 0.1 * self.model_rel as f32 * self.model_rel as f32
            {
                max_index = 1;
            }

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
}
pub fn decode_mask_to_palette(
    output: &SAM2Output,
    palette: &Arc<Mutex<palette::Palette>>,
    threshold: f32,
) {
    let mut palette = palette.lock().unwrap();
    let mut num_new_patches = 0;
    let is_point_segment = output.coordinates.shape()[0] == 1;
    for (mask_logit, coordinate) in output
        .mask_logits
        .axis_iter(Axis(0))
        .zip(output.coordinates.axis_iter(Axis(0)))
    {
        let mask = mask_logit.mapv(|x| (x > threshold) as usize);
        let coord_y = coordinate[0] as usize;
        let coord_x = coordinate[1] as usize;
        let patch_size = output.patch_size;
        let num_patches = palette.num_patches;

        let mut palette_slice = palette.map.slice_mut(s![
            coord_y..coord_y + patch_size,
            coord_x..coord_x + patch_size
        ]);

        // overlap detection. If the area already has a patch (by a overlap ratio), skip this patch
        // TODO: possible hyper-parameter
        // TODO: the strategy is not perfect. It may miss some patches
        if !is_point_segment {
            let exist_mask = palette_slice.mapv(|x| (x != 0) as usize) * &mask;
            let exist_ratio = exist_mask.sum() as f32 / mask.sum() as f32;
            if exist_ratio > 0.1 {
                continue;
            }
        }

        num_new_patches += 1;
        for ((y, x), &mask_val) in mask.indexed_iter() {
            if mask_val != 0 {
                palette_slice[[y, x]] = (num_patches + num_new_patches) as usize;
            }
        }
        let size = palette.size;
        palette.bboxes.push(get_bbox_from_mask(
            mask.view(),
            [coord_y, coord_x],
            patch_size,
            size,
        ));
        palette.valid.push(true);
    }
    palette.num_patches += num_new_patches;
}

/// Find the square bounding box for segmentation represented by the mask.
///
/// # Arguments
///
/// * `mask` - A 2D array view of usize representing the segmentation mask.
///            Values greater than 0 are considered part of the segmentation.
/// * `coord` - The coordinate [x, y] of the top-left corner of the patch in the original image.
/// * `patch_size` - The size of the patch (width and height).
/// * `size` - The size of original image.
///
/// # Returns
///
/// A 3-element array [x, y, size] representing the bounding box
/// of the segmentation in the original image coordinates.
fn get_bbox_from_mask(
    mask: ArrayView2<usize>,
    coord: [usize; 2],
    patch_size: usize,
    size: usize,
) -> [usize; 3] {
    let mut start_x_idx = patch_size;
    let mut start_y_idx = patch_size;
    let mut end_x_idx = 0;
    let mut end_y_idx = 0;
    for ((y, x), value) in mask.indexed_iter() {
        if *value > 0 {
            if x < start_x_idx {
                start_x_idx = x;
            }
            if x > end_x_idx {
                end_x_idx = x;
            }
            if y < start_y_idx {
                start_y_idx = y;
            }
            if y > end_y_idx {
                end_y_idx = y;
            }
        }
    }
    let width = end_x_idx - start_x_idx + 1;
    let height = end_y_idx - start_y_idx + 1;
    let mut start_y_coord = coord[0] + start_y_idx;
    let mut start_x_coord = coord[1] + start_x_idx;
    if width > height {
        let bbox_size = width;
        let height_offset = (width - height) / 2;
        start_y_coord = start_y_coord.saturating_sub(height_offset);
        if start_y_coord + bbox_size > size {
            start_y_coord = size - bbox_size;
        }
        return [start_x_coord, start_y_coord, bbox_size];
    } else {
        let bbox_size = height;
        let width_offset = (height - width) / 2;
        start_x_coord = start_x_coord.saturating_sub(width_offset);
        if start_x_coord + bbox_size > size {
            start_x_coord = size - bbox_size;
        }
        return [start_x_coord, start_y_coord, bbox_size];
    }
}
