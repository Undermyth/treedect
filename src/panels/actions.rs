use eframe::egui;
use indicatif::ProgressBar;
use ndarray::s;
use phf::{Map, phf_map};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::models::cluster;
use crate::models::dam2;
use crate::models::dinov2;
use crate::models::sam2;
use crate::panels::canvas;
use crate::panels::global;
use crate::panels::palette;
use crate::utils::score;

static MODEL2FILENAME: Map<&'static str, &'static str> = phf_map! {
    "sam2_small" => "sam2_hiera_small",
    "sam2_large" => "sam2_hiera_large",
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
pub fn save_csv_action(sender: Sender<String>) {
    let task = rfd::AsyncFileDialog::new()
        .add_filter("CSV Files", &["csv"])
        .save_file();
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
pub fn save_img_action(sender: Sender<String>) {
    let task = rfd::AsyncFileDialog::new()
        .add_filter("Image Files", &["tif", "tiff"])
        .save_file();
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

// pub fn load_depth_model_action(global: &mut global::GlobalState) {
//     let model_name = match global.params.depth_model_name.as_deref() {
//         Some(name) => name,
//         None => {
//             global.progress_state =
//                 global::ProgressState::Error("No depth model selected".to_string());
//             return;
//         }
//     };
//     let model_prefix = match MODEL2FILENAME.get(model_name) {
//         Some(prefix) => prefix,
//         None => {
//             global.progress_state =
//                 global::ProgressState::Error(format!("Unknown depth model: {model_name}"));
//             return;
//         }
//     };
//     let model_path = std::path::Path::new(&global.params.model_dir)
//         .join(format!("{}.onnx", model_prefix))
//         .to_string_lossy()
//         .to_string();
//     if !std::path::Path::new(&model_path).exists() {
//         global.progress_state =
//             global::ProgressState::Error(format!("Depth model not found: {}", model_path));
//         return;
//     }
//     let result = dam2::DAM2Model::from_path(518, &model_path);
//     match result {
//         Ok(model) => {
//             global.depth_model = Some(Arc::new(Mutex::new(model)));
//             global.progress_state =
//                 global::ProgressState::Finished(format!("Depth model {model_name} loaded"));
//         }
//         Err(e) => {
//             log::error!("Error when loading ONNX depth model: {e}");
//             global.progress_state =
//                 global::ProgressState::Error(format!("Error when loading ONNX depth model: {e}"));
//         }
//     }
// }

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
    let result = sam2::SAM2Model::from_path(
        // global.params.segment_rel as usize,
        1024,
        &encoder_path,
        &decoder_path,
    );
    match result {
        Ok(model) => {
            global.segment_model = Some(Arc::new(Mutex::new(model)));
            global.progress_state =
                global::ProgressState::Finished(format!("Segmentation model {model_name} loaded"));
        }
        Err(e) => {
            log::error!("Error when loading ONNX segmentation model: {e}");
            global.progress_state = global::ProgressState::Error(format!(
                "Error when loading ONNX segmentation model: {e}"
            ));
        }
    }
}

pub fn load_classify_model_action(global: &mut global::GlobalState) {
    let model_name = match global.params.classify_model_name.as_deref() {
        Some(name) => name,
        None => {
            global.progress_state =
                global::ProgressState::Error("No classification model selected".to_string());
            return;
        }
    };
    let model_prefix = match MODEL2FILENAME.get(model_name) {
        Some(prefix) => prefix,
        None => {
            global.progress_state =
                global::ProgressState::Error(format!("Unknown classification model: {model_name}"));
            return;
        }
    };
    let model_path = std::path::Path::new(&global.params.model_dir)
        .join(format!("{}.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    if !std::path::Path::new(&model_path).exists() {
        global.progress_state =
            global::ProgressState::Error(format!("Classification model not found: {}", model_path));
        return;
    }
    let result = dinov2::Dinov2Model::from_path(&model_path);
    match result {
        Ok(model) => {
            global.classify_model = Some(Arc::new(Mutex::new(model)));
            global.progress_state = global::ProgressState::Finished(format!(
                "Classification model {model_name} loaded"
            ));
        }
        Err(e) => {
            log::error!("Error when loading ONNX classification model: {e}");
            global.progress_state = global::ProgressState::Error(format!(
                "Error when loading ONNX classification model: {e}"
            ));
        }
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

            // // Green should be the dominant color
            // g > r && g > b
            true
        })
        .cloned()
        .collect()
}

// pub fn haware_sampling_action(
//     global: &mut global::GlobalState,
//     progress_sender: Sender<f32>,
//     depth_sender: Sender<Vec<[usize; 2]>>,
// ) {
//     let raw_image = global.raw_image.as_ref().unwrap().clone();
//     let batcher = dam2::DAM2Batcher::new(global.params.segment_rel as usize, raw_image);
//     let dilation_radius = global.params.dilation_radius;
//     let nms_radius = global.params.nms_radius;
//     let model = global.depth_model.as_mut().unwrap().clone();
//     std::thread::spawn(move || {
//         let total_length = batcher.len();
//         let mut sampling_points = Vec::<[usize; 2]>::new();
//         let bar = ProgressBar::new(total_length as u64);
//         bar.set_style(
//             indicatif::ProgressStyle::with_template(
//                 "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
//             )
//             .unwrap(),
//         );
//         bar.set_prefix("Depth Estimation");
//         for batch in batcher.into_iter() {
//             let result = model.lock().unwrap().forward(batch);
//             match result {
//                 Ok(result) => {
//                     let samples = dam2::decode_depth_to_sample(&result, dilation_radius);
//                     sampling_points.extend(samples);
//                 }
//                 Err(e) => {
//                     log::error!("Error: {e}");
//                 }
//             }
//             bar.inc(1);
//             progress_sender
//                 .send(bar.position() as f32 / total_length as f32)
//                 .unwrap();
//         }
//         let sampling_points = dam2::filter_near_samples(sampling_points, nms_radius);
//         depth_sender.send(sampling_points).unwrap();
//     });
//     // sampling_points
// }

pub fn segment_action(
    global: &mut global::GlobalState,
    progress_sender: Sender<f32>,
    segment_sender: Sender<palette::Palette>,
) {
    let raw_image = global.raw_image.as_ref().unwrap();
    let raw_image = raw_image.clone();
    let patch_size = global.params.segment_rel;
    let lumin_filt = global.params.luminance_filt;
    let x_scan_interval = global.params.x_scan_interval;
    let y_scan_interval = global.params.y_scan_interval;
    let merge_thr = global.params.merge_thr;

    let model = global.segment_model.as_mut().unwrap().clone();
    std::thread::spawn(move || {
        let palette = model.lock().unwrap().tiled_diffuse_merge_scan(
            raw_image,
            patch_size as usize,
            lumin_filt,
            x_scan_interval,
            y_scan_interval,
            merge_thr,
        );
        segment_sender.send(palette).unwrap();
    });
}

pub fn point_segment_action(global: &mut global::GlobalState) {
    let palette = global.palette.as_ref().unwrap();
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
    // let mask_threshold = global.params.mask_threshold;
    let mask_threshold = 0.0;
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
                let mask = result
                    .mask_logits
                    .slice(s![0, .., ..])
                    .mapv(|x| (x > 0.0) as usize);
                palette.lock().unwrap().add_segment(
                    mask,
                    [result.coordinates[(0, 0)], result.coordinates[(0, 1)]],
                    result.patch_size,
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

pub fn classify_action(
    global: &mut global::GlobalState,
    progress_sender: Sender<f32>,
    classify_sender: Sender<bool>,
) {
    let raw_image = global.raw_image.as_ref().unwrap();
    let raw_image = raw_image.clone();
    let palette = global.palette.as_ref().unwrap();
    let mut batcher =
        dinov2::Dinov2Batcher::new(global.params.batch_size, raw_image, palette.clone());
    let palette = palette.clone();
    let length = batcher.len();
    let model = global.classify_model.as_mut().unwrap().clone();
    let n_classes = global.params.n_classes;
    std::thread::spawn(move || {
        let bar = ProgressBar::new(length as u64);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
            )
            .unwrap(),
        );
        bar.set_prefix("Feature Extraction");
        let mut features: Option<dinov2::Dinov2Output> = None;
        for batch in batcher.into_iter() {
            let actual_batch_size = batch.image.shape()[0];
            let result = model.lock().unwrap().forward(batch);
            match result {
                Ok(output) => {
                    assert!(
                        !output.features.is_any_nan(),
                        "NaN detected in extracted features"
                    );
                    if features.is_none() {
                        features = Some(output);
                    } else {
                        features.as_mut().unwrap().concat(&output);
                    }
                }
                Err(e) => {
                    log::error!("Error: {e}");
                }
            }
            bar.inc(actual_batch_size as u64);
            let percent = bar.position() as f32 / length as f32;
            progress_sender.send(percent).unwrap();
        }
        bar.finish();
        // palette.lock().unwrap().get_statistics();
        let features = parse_features(features.unwrap(), palette.clone());
        assert!(!features.features.is_any_nan(), "NaN detected in features");
        let output = cluster::cluster(features, n_classes);
        palette.lock().unwrap().set_cluster_map(output);

        // log::info!("{:?}", output);
        classify_sender.send(true).unwrap();
    });
}

fn parse_features(
    features: dinov2::Dinov2Output,
    palette: Arc<Mutex<palette::Palette>>,
) -> cluster::SegmentFeatures {
    let mut valid_areas = Vec::new();
    let palette = palette.lock().unwrap();
    for (index, valid) in palette.valid.iter().enumerate() {
        if *valid {
            valid_areas.push(palette.areas[index]);
        }
    }
    assert!(valid_areas.len() == features.segment_ids.len());
    cluster::SegmentFeatures {
        segment_ids: features.segment_ids,
        features: features.features,
        areas: valid_areas,
    }
}

pub fn get_importance_score(global: &mut global::GlobalState) {
    let palette = global.palette.as_ref().unwrap();
    let palette = palette.clone();
    let table = score::Table::build_from_palette(palette);
    global.score_table = Some(table);
}

pub fn export_image_action(global: &mut global::GlobalState, path: String) {
    // 1. 获取图像尺寸（从第一个可见图层）
    let (width, height) = {
        let first_visible = global.layers.iter().find(|l| l.visible);
        match first_visible {
            Some(layer) => {
                let size = layer.get_image_size();
                (size[0], size[1])
            }
            None => {
                log::error!("No visible layers to export");
                global.progress_state =
                    global::ProgressState::Error("No visible layers to export".to_string());
                return;
            }
        }
    };

    // 2. 创建RGBA缓冲区（支持透明背景）
    let mut output_image = image::RgbaImage::new(width as u32, height as u32);

    // 3. 遍历所有可见图层并叠加
    for layer in &global.layers {
        if !layer.visible || layer.opacity <= 0.0 {
            continue;
        }

        // 从raw_image获取RGB数据
        let Some(ref raw_image) = layer.raw_image else {
            continue;
        };

        // 根据LayerImage类型提取像素数据
        match raw_image {
            canvas::LayerImage::RGBImage(img) => {
                blend_rgb_image(&mut output_image, img, layer.opacity);
            }
            canvas::LayerImage::RGBAImage(img) => {
                blend_rgba_image(&mut output_image, img, layer.opacity);
            }
            canvas::LayerImage::EguiImage(img) => {
                blend_egui_image(&mut output_image, img, layer.opacity);
            }
        }
    }

    // 4. 添加文字层（如果show_cluster_ids=true且palette存在且有clusters）
    if global.params.show_cluster_ids {
        if let Some(ref palette_arc) = global.palette {
            let palette = palette_arc.lock().unwrap();
            if palette.num_clusters > 0 {
                // 加载嵌入字体
                let font_data = include_bytes!("../IosevkaTermNerdFont-Regular.ttf");
                let font = ab_glyph::FontArc::try_from_slice(font_data)
                    .expect("Failed to load embedded font");

                // 字体大小固定为20
                let font_size = 200.0;
                let scale = ab_glyph::PxScale {
                    x: font_size,
                    y: font_size,
                };

                // 绘制每个cluster的文字
                for (i, cluster_id) in palette.cluster_map.iter().enumerate() {
                    let [x, y, size] = palette.bboxes[i];
                    let center_x = (x + size / 2) as i32;
                    let center_y = (y + size / 2) as i32;

                    let text = cluster_id.to_string();

                    // 使用imageproc绘制白色文字
                    let color = image::Rgba([255, 255, 255, 255]);
                    output_image = imageproc::drawing::draw_text(
                        &output_image,
                        color,
                        center_x,
                        center_y,
                        scale,
                        &font,
                        &text,
                    );
                }
            }
        }
    }

    // 5. 保存为TIFF文件
    let path = std::path::Path::new(&path);
    match std::fs::File::create(path) {
        Ok(file) => {
            let mut encoder =
                tiff::encoder::TiffEncoder::new(file).expect("Failed to create TIFF encoder");

            // 将RGBA数据转换为RGB数据（TIFF通常不保存alpha通道，但我们可以尝试保存带透明度的）
            // 或者直接保存RGBA
            let width = output_image.width();
            let height = output_image.height();
            let data = output_image.into_raw();

            if let Err(e) =
                encoder.write_image::<tiff::encoder::colortype::RGBA8>(width, height, &data)
            {
                log::error!("Failed to write TIFF file: {}", e);
                global.progress_state =
                    global::ProgressState::Error(format!("Failed to write TIFF file: {}", e));
            } else {
                log::info!("Image exported successfully to: {:?}", path);
                global.progress_state =
                    global::ProgressState::Finished(format!("Image exported to: {:?}", path));
            }
        }
        Err(e) => {
            log::error!("Failed to create output file: {}", e);
            global.progress_state =
                global::ProgressState::Error(format!("Failed to create output file: {}", e));
        }
    }
}

/// 将RGB图像混合到输出缓冲区
fn blend_rgb_image(output: &mut image::RgbaImage, input: &image::RgbImage, opacity: f32) {
    let (width, height) = (output.width() as usize, output.height() as usize);
    let input_width = input.width() as usize;
    let input_height = input.height() as usize;

    for y in 0..height.min(input_height) {
        for x in 0..width.min(input_width) {
            let src_pixel = input.get_pixel(x as u32, y as u32);
            let dst_pixel = output.get_pixel_mut(x as u32, y as u32);

            // 应用opacity进行alpha混合
            let alpha = opacity;
            dst_pixel[0] = blend_channel(dst_pixel[0], src_pixel[0], alpha);
            dst_pixel[1] = blend_channel(dst_pixel[1], src_pixel[1], alpha);
            dst_pixel[2] = blend_channel(dst_pixel[2], src_pixel[2], alpha);
            dst_pixel[3] = (255.0 * alpha + dst_pixel[3] as f32 * (1.0 - alpha)) as u8;
        }
    }
}

/// 将RGBA图像混合到输出缓冲区
fn blend_rgba_image(output: &mut image::RgbaImage, input: &image::RgbaImage, opacity: f32) {
    let (width, height) = (output.width() as usize, output.height() as usize);
    let input_width = input.width() as usize;
    let input_height = input.height() as usize;

    for y in 0..height.min(input_height) {
        for x in 0..width.min(input_width) {
            let src_pixel = input.get_pixel(x as u32, y as u32);
            let dst_pixel = output.get_pixel_mut(x as u32, y as u32);

            // 应用opacity和源alpha进行混合
            let src_alpha = (src_pixel[3] as f32 / 255.0) * opacity;
            dst_pixel[0] = blend_channel(dst_pixel[0], src_pixel[0], src_alpha);
            dst_pixel[1] = blend_channel(dst_pixel[1], src_pixel[1], src_alpha);
            dst_pixel[2] = blend_channel(dst_pixel[2], src_pixel[2], src_alpha);
            dst_pixel[3] = (src_alpha * 255.0 + dst_pixel[3] as f32 * (1.0 - src_alpha)) as u8;
        }
    }
}

/// 将egui ColorImage混合到输出缓冲区
fn blend_egui_image(output: &mut image::RgbaImage, input: &egui::ColorImage, opacity: f32) {
    let (width, height) = (output.width() as usize, output.height() as usize);
    let [input_width, input_height] = input.size;

    for y in 0..height.min(input_height) {
        for x in 0..width.min(input_width) {
            let src_pixel = input[(x, y)];
            let dst_pixel = output.get_pixel_mut(x as u32, y as u32);

            // 应用opacity和源alpha进行混合
            let src_alpha = (src_pixel.a() as f32 / 255.0) * opacity;
            dst_pixel[0] = blend_channel(dst_pixel[0], src_pixel.r(), src_alpha);
            dst_pixel[1] = blend_channel(dst_pixel[1], src_pixel.g(), src_alpha);
            dst_pixel[2] = blend_channel(dst_pixel[2], src_pixel.b(), src_alpha);
            dst_pixel[3] = (src_alpha * 255.0 + dst_pixel[3] as f32 * (1.0 - src_alpha)) as u8;
        }
    }
}

/// 混合单个颜色通道
fn blend_channel(dst: u8, src: u8, alpha: f32) -> u8 {
    (src as f32 * alpha + dst as f32 * (1.0 - alpha)) as u8
}
