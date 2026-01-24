use fast_image_resize as fr;
use fast_image_resize::images::Image;
use image::{DynamicImage, GenericImageView, Luma};
use image::{ImageBuffer, RgbImage};
use imageproc::morphology;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, Ix3, array, s};
use ndarray_stats::QuantileExt;
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use std::sync::{Arc, Mutex};

pub struct DAM2Batch {
    // [1, C, H, W] where H = W = 518
    pub image: Array4<f32>,
    /// [2], left up corner [x, y] of the patches.
    /// The coordinate is in the format [width, height], aligned with the image coordinate system.
    /// The whole patch will be [x..x+patch_size, y..y+patch_size]
    pub coordinates: Array1<usize>,
    /// original size of patch
    pub original_size: usize,
}

impl DAM2Batch {
    #[allow(dead_code)]
    pub fn print(&self) {
        log::info!("Image shape: {:?}", self.image.shape());
        log::info!("Coordinates: {:?}", self.coordinates);
        log::info!("Original size: {}", self.original_size);
    }
}

pub struct DAM2Batcher {
    x_pos: usize,
    y_pos: usize,
    length: usize,
    patch_size: usize,
    model_rel: usize,
    raw_image: Arc<Mutex<RgbImage>>,
    mean: Array1<f32>,
    std: Array1<f32>,
}

impl DAM2Batcher {
    pub fn new(patch_size: usize, raw_image: Arc<Mutex<RgbImage>>) -> Self {
        let width = raw_image.lock().unwrap().width() as usize;
        let height = raw_image.lock().unwrap().height() as usize;
        let stride = patch_size / 2;
        let n_width_patch = width / stride - 1 + (width % stride != 0) as usize;
        let n_height_patch = height / stride - 1 + (height % stride != 0) as usize;
        Self {
            x_pos: 0,
            y_pos: 0,
            length: n_width_patch * n_height_patch,
            patch_size,
            model_rel: 518,
            raw_image,
            mean: Array1::from_vec(vec![0.485, 0.456, 0.406]),
            std: Array1::from_vec(vec![0.229, 0.224, 0.225]),
        }
    }
    pub fn len(&self) -> usize {
        self.length
    }
}

impl Iterator for DAM2Batcher {
    type Item = DAM2Batch;
    fn next(&mut self) -> Option<Self::Item> {
        let raw_image = self.raw_image.lock().unwrap();
        if self.y_pos == raw_image.height() as usize {
            return None;
        }
        let patch = raw_image
            .view(
                self.x_pos as u32,
                self.y_pos as u32,
                self.patch_size as u32,
                self.patch_size as u32,
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
        sample = sample / 255.0
            - self.mean.to_shape([1, 1, 3]).unwrap() / self.std.to_shape([1, 1, 3]).unwrap();

        let batch = DAM2Batch {
            image: sample
                .to_shape([1, self.model_rel, self.model_rel, 3])
                .unwrap()
                .permuted_axes([0, 3, 1, 2])
                .as_standard_layout()
                .to_owned(),
            coordinates: array![self.x_pos, self.y_pos].to_owned(),
            original_size: self.patch_size,
        };

        let stride = self.patch_size / 2;
        self.x_pos += stride;
        if self.x_pos + stride == raw_image.width() as usize {
            self.x_pos = 0;
            self.y_pos += stride;
            if self.y_pos + stride == raw_image.height() as usize {
                self.y_pos = raw_image.height() as usize;
            } else if self.y_pos + self.patch_size > raw_image.height() as usize {
                self.y_pos = raw_image.height() as usize - self.patch_size;
            }
        } else if self.x_pos + self.patch_size > raw_image.width() as usize {
            self.x_pos = raw_image.width() as usize - self.patch_size;
        }

        // batch.print();
        // self.y_pos = self.raw_image.height() as usize;
        Some(batch)
    }
}
pub struct DAM2Model {
    pub session: Session,
    pub model_rel: usize,
}

pub struct DAM2Output {
    pub depth_logits: Array2<f32>,
    pub coordinates: Array1<usize>,
    pub patch_size: usize,
}

impl DAM2Model {
    pub fn from_path(
        model_rel: usize,
        path: &str,
        initialize: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if initialize {
            log::info!("Initializing ONNX Runtime with execution provider");
            ort::init()
                .with_execution_providers([DirectMLExecutionProvider::default()
                    .build()
                    .error_on_failure()])
                .commit()?;
        }
        log::info!("Loading model from: {}", path);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;
        log::info!("Successfully loaded model");
        log::info!("Intrinsic resolution of the model: {}", model_rel);
        Ok(Self { model_rel, session })
    }

    pub fn forward(
        &mut self,
        batch_image: DAM2Batch,
    ) -> Result<DAM2Output, Box<dyn std::error::Error>> {
        let input_tensor = TensorRef::from_array_view(&batch_image.image)?;
        let output = self.session.run(ort::inputs!["l_x_" => input_tensor])?;
        let depth_logits = &output["select_36"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?;
        let depth_logits = depth_logits.slice(s![0, .., ..]);
        let depth_logits = depth_logits.as_standard_layout().into_owned();
        let (depth_vec, _) = depth_logits.into_raw_vec_and_offset();
        let depth_bytes: Vec<u8> = bytemuck::cast_slice(&depth_vec).to_vec();
        let depth_image = fr::images::Image::from_vec_u8(
            self.model_rel as u32,
            self.model_rel as u32,
            depth_bytes,
            fr::PixelType::F32,
        );
        let mut scaled_depth = fr::images::Image::new(
            batch_image.original_size as u32,
            batch_image.original_size as u32,
            fr::PixelType::F32,
        );
        let mut resizer = fr::Resizer::new();
        let result = resizer.resize(depth_image.as_ref().unwrap(), &mut scaled_depth, None);
        if let Err(e) = result {
            log::error!("Error resizing depth: {:?}", e);
        }
        let scaled_depth = scaled_depth.into_vec();
        let scaled_depth: &[f32] = bytemuck::cast_slice(&scaled_depth);
        let scaled_depth = scaled_depth.to_vec();
        let scaled_depth = ArrayView2::from_shape(
            (batch_image.original_size, batch_image.original_size),
            scaled_depth.as_slice(),
        )?;
        Ok(DAM2Output {
            depth_logits: scaled_depth.to_owned(),
            coordinates: batch_image.coordinates,
            patch_size: batch_image.original_size,
        })
    }
}

pub fn decode_depth_to_sample(output: &DAM2Output, dilation_radius: usize) -> Vec<[usize; 2]> {
    let minimum = *(output.depth_logits.view().min().unwrap());
    let maximum = *(output.depth_logits.view().max().unwrap());
    let raw_data: Vec<u8> = output
        .depth_logits
        .iter()
        .map(|&val| ((val - minimum) / (maximum - minimum) * 255.0) as u8)
        .collect();
    let image: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(output.patch_size as u32, output.patch_size as u32, raw_data)
            .unwrap();
    // let path = std::path::Path::new("dilated_image.png");
    // image.save(path).unwrap();
    let dilated_image =
        morphology::grayscale_dilate(&image, &morphology::Mask::disk(dilation_radius as u8));
    let mut sampling_points = Vec::new();
    for (x, y, pixel) in image.enumerate_pixels() {
        // skip edge pixels
        let padding = 10;
        if x < padding
            || x >= image.width() - padding
            || y < padding
            || y >= image.height() - padding
        {
            continue;
        }
        let original_val = pixel[0];
        let dilated_val = dilated_image.get_pixel(x, y)[0];

        // 核心逻辑：如果原值等于膨胀后的值，说明它是局部最大
        // 设置一个微小的阈值以过滤纯黑区域
        if original_val > 0 && original_val == dilated_val {
            sampling_points.push([
                x as usize + output.coordinates[0],
                y as usize + output.coordinates[1],
            ]);
        }
    }
    sampling_points
}

pub fn filter_near_samples(sampling_points: Vec<[usize; 2]>, radius: usize) -> Vec<[usize; 2]> {
    if radius == 0 {
        return sampling_points;
    }

    let mut result = Vec::new();
    // Use a spatial hash grid to efficiently check for nearby points.
    // Key: (grid_x, grid_y), Value: [point_x, point_y]
    let mut occupied_cells: std::collections::HashMap<(i32, i32), [usize; 2]> =
        std::collections::HashMap::new();
    let radius_sq = (radius as u64) * (radius as u64);

    for point in sampling_points {
        let x = point[0];
        let y = point[1];
        let grid_x = (x / radius) as i32;
        let grid_y = (y / radius) as i32;

        let mut is_too_close = false;

        // Check the 3x3 neighborhood around the current grid cell
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(&[qx, qy]) = occupied_cells.get(&(grid_x + dx, grid_y + dy)) {
                    let dx_val = x as i64 - qx as i64;
                    let dy_val = y as i64 - qy as i64;
                    let dist_sq = (dx_val * dx_val + dy_val * dy_val) as u64;

                    if dist_sq < radius_sq {
                        is_too_close = true;
                        break;
                    }
                }
            }
            if is_too_close {
                break;
            }
        }

        if !is_too_close {
            occupied_cells.insert((grid_x, grid_y), [x, y]);
            result.push([x, y]);
        }
    }

    result
}
