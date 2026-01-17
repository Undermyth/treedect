use eframe::egui;
use indicatif::ProgressBar;
use phf::{Map, phf_map};
use std::sync::mpsc::Sender;

use crate::models::dam2;
use crate::models::dinov2::Dinov2Model;
use crate::models::sam2;
use crate::panels::canvas;
use crate::panels::global;

static MODEL2FILENAME: Map<&'static str, &'static str> = phf_map! {
    "sam2_small" => "sam2_hiera_small",
    "dinov2_base" => "dinov2_vitb_reg",
    "depv2_base" => "depth_anything_v2_vitb",
};

pub fn load_image_action(ctx: egui::Context, sender: Sender<String>) {
    let task = rfd::AsyncFileDialog::new()
        .add_filter("Image Files", &["jpg", "png", "jpeg", "bmp", "tif", "tiff"])
        .pick_file();
    std::thread::spawn(move || {
        let future = async move {
            if let Some(file_handle) = task.await {
                if let Some(path) = file_handle.path().to_str() {
                    let _ = sender.send(path.to_string());
                    ctx.request_repaint();
                }
            }
        };
        futures::executor::block_on(future);
    });
}

pub fn select_model_path_action(sender: Sender<String>) {
    let task = rfd::AsyncFileDialog::new().pick_folder();
    std::thread::spawn(move || {
        let future = async move {
            if let Some(file_handle) = task.await {
                if let Some(path) = file_handle.path().to_str() {
                    let _ = sender.send(path.to_string());
                }
            }
        };
        futures::executor::block_on(future);
    });
}

pub fn load_depth_model_action(global: &mut global::GlobalState) {
    let model_prefix = MODEL2FILENAME
        .get(global.params.depth_model_name.as_deref().unwrap_or(""))
        .unwrap();
    let model_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let mut initialize = false;
    if !global.ort_initialized {
        initialize = true;
        global.ort_initialized = true;
    }
    let result = dam2::DAM2Model::from_path(518, &model_path, initialize);
    if let Ok(model) = result {
        global.depth_model = Some(model);
    }
}

pub fn load_segment_model_action(global: &mut global::GlobalState) {
    let model_prefix = MODEL2FILENAME
        .get(global.params.segment_model_name.as_deref().unwrap_or(""))
        .unwrap();
    let encoder_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.encoder.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let decoder_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.decoder.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let mut initialize = false;
    if !global.ort_initialized {
        initialize = true;
        global.ort_initialized = true;
    }
    let result = sam2::SAM2Model::from_path(
        // global.params.segment_rel as usize,
        1024,
        &encoder_path,
        &decoder_path,
        initialize,
    );
    match result {
        Ok(model) => {
            global.segment_model = Some(model);
        }
        Err(e) => {
            log::error!("Error: {e}");
        }
    }
}

pub fn load_classify_model_action(global: &mut global::GlobalState) {
    let model_prefix = MODEL2FILENAME
        .get(global.params.classify_model_name.as_deref().unwrap_or(""))
        .unwrap();
    let model_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let mut initialize = false;
    if !global.ort_initialized {
        initialize = true;
        global.ort_initialized = true;
    }
    let result = Dinov2Model::from_path(&model_path, initialize);
    if let Ok(model) = result {
        global.classify_model = Some(model);
    }
}

/// Generates a grid of sampling points for image processing.
///
/// This function creates a uniform grid of points spaced at regular intervals across
/// a given width and height. The points are distributed symmetrically and centered
/// within the specified dimensions.
///
/// # Arguments
///
/// * `interval` - The desired spacing between sampling points
/// * `width` - The width of the area to sample
/// * `height` - The height of the area to sample
///
/// # Returns
///
/// A vector of 2-element arrays containing [x, y] coordinates of the sampling points.
/// [x, y] is in the format [width, height], aligned with the image coordinate system.
pub fn grid_sampling_action(interval: usize, width: usize, height: usize) -> Vec<[usize; 2]> {
    let mut sampling_points = Vec::<[usize; 2]>::new();

    // Calculate padding to ensure symmetric spacing
    // The idea is to distribute the points evenly across the width and height
    let num_x_points = (width + interval - 1) / interval; // Ceiling division
    let num_y_points = (height + interval - 1) / interval; // Ceiling division

    // Calculate actual spacing to ensure symmetric distribution
    let actual_x_spacing = if num_x_points > 1 {
        width / (num_x_points - 1)
    } else {
        0
    };
    let actual_y_spacing = if num_y_points > 1 {
        height / (num_y_points - 1)
    } else {
        0
    };

    // Calculate starting offset to center the grid
    let start_x = if num_x_points > 1 {
        (width - (num_x_points - 1) * actual_x_spacing) / 2
    } else {
        width / 2
    };
    let start_y = if num_y_points > 1 {
        (height - (num_y_points - 1) * actual_y_spacing) / 2
    } else {
        height / 2
    };

    for y_idx in 0..num_y_points {
        for x_idx in 0..num_x_points {
            let x = start_x + x_idx * actual_x_spacing;
            let y = start_y + y_idx * actual_y_spacing;

            // Ensure coordinates are within bounds
            if x < width && y < height {
                sampling_points.push([x, y]);
            }
        }
    }

    sampling_points
}

pub fn filter_sampling_action(
    sampling_points: &mut Vec<[usize; 2]>,
    image: &canvas::LayerImage,
) -> Vec<[usize; 2]> {
    sampling_points
        .iter()
        .filter(|point| {
            let [x, y] = point;
            let pixel = image.get_pixel(*x, *y);
            let avg_rgb = (pixel.r as u32 + pixel.g as u32 + pixel.b as u32) / 3;

            // Check if the point is bright enough
            if avg_rgb < 60 {
                return false;
            }

            // Check if the point is roughly green (green component should be significantly higher than red and blue)
            let r = pixel.r as u32;
            let g = pixel.g as u32;
            let b = pixel.b as u32;

            // Green should be the dominant color
            g > r && g > b
        })
        .cloned()
        .collect()
}

pub fn haware_sampling_action(global: &mut global::GlobalState) -> Vec<[usize; 2]> {
    let mut sampling_points = Vec::<[usize; 2]>::new();
    let raw_image = global.raw_image.as_ref().unwrap();
    if let canvas::LayerImage::RGBImage(image) = raw_image {
        let batcher = dam2::DAM2Batcher::new(global.params.segment_rel as usize, image);
        let bar = ProgressBar::new(batcher.len() as u64);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
            )
            .unwrap(),
        );
        bar.set_prefix("Segmenting");
        for batch in batcher.into_iter() {
            let result = global.depth_model.as_mut().unwrap().forward(batch);
            match result {
                Ok(result) => {
                    let samples =
                        dam2::decode_depth_to_sample(&result, global.params.dilation_radius);
                    sampling_points.extend(samples);
                }
                Err(e) => {
                    log::error!("Error: {e}");
                }
            }
            bar.inc(1);
        }
    }
    dam2::filter_near_samples(sampling_points, global.params.nms_radius)
    // sampling_points
}

pub fn segment_action(global: &mut global::GlobalState, palette: Option<&mut canvas::Palette>) {
    let mut palette = match palette {
        Some(palette) => palette,
        None => global.layers[2].palette.as_mut().unwrap(),
    };
    let sampling_points = global.sampling_points.clone().unwrap();
    let raw_image = global.raw_image.as_ref().unwrap();
    let bar = ProgressBar::new(sampling_points.len() as u64);
    if let canvas::LayerImage::RGBImage(image) = raw_image {
        let batcher = sam2::SAM2Batcher::new(
            global.params.batch_size,
            global.params.segment_rel as usize,
            sampling_points,
            image,
        );
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
            )
            .unwrap(),
        );
        bar.set_prefix("Segmenting");
        for batch in batcher.into_iter() {
            // let start_time = std::time::Instant::now();
            let actual_batch_size = batch.image.shape()[0];
            let result = global.segment_model.as_mut().unwrap().forward(batch);
            match result {
                Ok(result) => {
                    global
                        .segment_model
                        .as_ref()
                        .unwrap()
                        .decode_mask_to_palette(
                            &result,
                            &mut palette,
                            global.params.mask_threshold,
                        );
                }
                Err(e) => {
                    log::error!("Error: {e}");
                }
            }
            bar.inc(actual_batch_size as u64);
            // let end_time = std::time::Instant::now();
            // log::info!("Inference time taken: {:?}", end_time - start_time);
        }
        bar.finish();
    }
}
