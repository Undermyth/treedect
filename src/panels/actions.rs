use eframe::egui;
use indicatif::ProgressBar;
use phf::{Map, phf_map};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::models::dam2;
use crate::models::dinov2::Dinov2Model;
use crate::models::sam2;
use crate::panels::global;
use crate::panels::palette;

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
    let model_name = match global.params.depth_model_name.as_deref() {
        Some(name) => name,
        None => {
            global.progress_state =
                global::ProgressState::Error("No depth model selected".to_string());
            return;
        }
    };
    let model_prefix = match MODEL2FILENAME.get(model_name) {
        Some(prefix) => prefix,
        None => {
            global.progress_state =
                global::ProgressState::Error(format!("Unknown depth model: {model_name}"));
            return;
        }
    };
    let model_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    if !std::path::Path::new(&model_path).exists() {
        global.progress_state =
            global::ProgressState::Error(format!("Depth model not found: {}", model_path));
        return;
    }
    let mut initialize = false;
    if !global.ort_initialized {
        initialize = true;
        global.ort_initialized = true;
    }
    let result = dam2::DAM2Model::from_path(518, &model_path, initialize);
    match result {
        Ok(model) => {
            global.depth_model = Some(Arc::new(Mutex::new(model)));
        }
        Err(e) => {
            log::error!("Error when loading ONNX depth model: {e}");
        }
    }
}

pub fn load_segment_model_action(global: &mut global::GlobalState) {
    let model_name = match global.params.segment_model_name.as_deref() {
        Some(name) => name,
        None => {
            global.progress_state =
                global::ProgressState::Error("No segmentation model selected".to_string());
            return;
        }
    };
    let model_prefix = match MODEL2FILENAME.get(model_name) {
        Some(prefix) => prefix,
        None => {
            global.progress_state =
                global::ProgressState::Error(format!("Unknown segmentation model: {model_name}"));
            return;
        }
    };
    let encoder_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.encoder.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let decoder_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.decoder.onnx", model_prefix))
        .to_string_lossy()
        .to_string();

    if !std::path::Path::new(&encoder_path).exists() {
        global.progress_state = global::ProgressState::Error(format!(
            "Segmentation encoder model not found: {}",
            encoder_path
        ));
        return;
    }

    if !std::path::Path::new(&decoder_path).exists() {
        global.progress_state = global::ProgressState::Error(format!(
            "Segmentation decoder model not found: {}",
            decoder_path
        ));
        return;
    }

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
            global.segment_model = Some(Arc::new(Mutex::new(model)));
        }
        Err(e) => {
            log::error!("Error when loading ONNX segmentation model: {e}");
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
    image: MutexGuard<image::RgbImage>,
) -> Vec<[usize; 2]> {
    sampling_points
        .iter()
        .filter(|point| {
            let [x, y] = point;
            let pixel = image[(*x as u32, *y as u32)];
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            let avg_rgb = (r + g + b) / 3;

            // Check if the point is bright enough
            if avg_rgb < 60 {
                return false;
            }

            // Green should be the dominant color
            g > r && g > b
        })
        .cloned()
        .collect()
}

pub fn haware_sampling_action(
    global: &mut global::GlobalState,
    progress_sender: Sender<f32>,
    depth_sender: Sender<Vec<[usize; 2]>>,
) {
    let raw_image = global.raw_image.as_ref().unwrap().clone();
    let batcher = dam2::DAM2Batcher::new(global.params.segment_rel as usize, raw_image);
    let dilation_radius = global.params.dilation_radius;
    let nms_radius = global.params.nms_radius;
    let model = global.depth_model.as_mut().unwrap().clone();
    std::thread::spawn(move || {
        let total_length = batcher.len();
        let mut sampling_points = Vec::<[usize; 2]>::new();
        let bar = ProgressBar::new(total_length as u64);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
            )
            .unwrap(),
        );
        bar.set_prefix("Depth Estimation");
        for batch in batcher.into_iter() {
            let result = model.lock().unwrap().forward(batch);
            match result {
                Ok(result) => {
                    let samples = dam2::decode_depth_to_sample(&result, dilation_radius);
                    sampling_points.extend(samples);
                }
                Err(e) => {
                    log::error!("Error: {e}");
                }
            }
            bar.inc(1);
            progress_sender
                .send(bar.position() as f32 / total_length as f32)
                .unwrap();
        }
        let sampling_points = dam2::filter_near_samples(sampling_points, nms_radius);
        depth_sender.send(sampling_points).unwrap();
    });
    // sampling_points
}

pub fn segment_action(
    global: &mut global::GlobalState,
    progress_sender: Sender<f32>,
    segment_sender: Sender<palette::Palette>,
) {
    let sampling_points = global.sampling_points.as_ref().unwrap().clone();
    let raw_image = global.raw_image.as_ref().unwrap();
    let width = raw_image.lock().unwrap().width() as usize;
    let palette = palette::Palette::new(width);
    let raw_image = raw_image.clone();
    let batcher = sam2::SAM2Batcher::new(
        global.params.batch_size,
        global.params.segment_rel as usize,
        sampling_points.clone(),
        raw_image,
    );
    let model = global.segment_model.as_mut().unwrap().clone();
    let mask_threshold = global.params.mask_threshold;
    std::thread::spawn(move || {
        let palette = Arc::new(Mutex::new(palette)); // no competition here. only for interface compatibility
        let bar = ProgressBar::new(sampling_points.len() as u64);
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
            let result = model.lock().unwrap().forward(batch);
            match result {
                Ok(result) => {
                    sam2::decode_mask_to_palette(&result, &palette, mask_threshold);
                }
                Err(e) => {
                    log::error!("Error: {e}");
                }
            }
            bar.inc(actual_batch_size as u64);
            let percent = bar.position() as f32 / sampling_points.len() as f32;
            progress_sender.send(percent).unwrap();
            // let end_time = std::time::Instant::now();
            // log::info!("Inference time taken: {:?}", end_time - start_time);
        }
        bar.finish();
        // unwrap the palette from mutex
        let mutex = Arc::try_unwrap(palette).unwrap();
        let palette = mutex.into_inner().unwrap();
        segment_sender.send(palette).unwrap();
    });
}

pub fn point_segment_action(global: &mut global::GlobalState) {
    let palette = global.layers[2].palette.as_ref().unwrap();
    let palette = palette.clone();
    let sampling_points = global.sampling_points.as_ref().unwrap().clone();
    let raw_image = global.raw_image.as_ref().unwrap();
    let raw_image = raw_image.clone();
    let batcher = sam2::SAM2Batcher::new(
        global.params.batch_size,
        global.params.segment_rel as usize,
        sampling_points.clone(),
        raw_image,
    );
    let model = global.segment_model.as_mut().unwrap().clone();
    let mask_threshold = global.params.mask_threshold;
    let bar = ProgressBar::new(sampling_points.len() as u64);
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
        let result = model.lock().unwrap().forward(batch);
        match result {
            Ok(result) => {
                sam2::decode_mask_to_palette(&result, &palette, mask_threshold);
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
