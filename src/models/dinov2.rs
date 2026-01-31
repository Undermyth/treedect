use fast_image_resize as fr;
use fast_image_resize::images::Image;
use image::RgbImage;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array3, Array4, Axis, Ix3, concatenate, s};
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use std::sync::{Arc, Mutex};

use crate::panels::palette;

pub struct Dinov2Output {
    pub segment_ids: Vec<usize>, // B ids, start from 1
    pub features: Array2<f32>,   // [B, d]
}

impl Dinov2Output {
    pub fn concat(&mut self, other: &Self) {
        self.segment_ids.extend(other.segment_ids.iter());
        self.features = concatenate![Axis(0), self.features, other.features];
    }
}

pub struct Dinov2Batch {
    pub image: Array4<f32>,         // [B, 3, 448, 448]
    pub token_ids: Vec<Vec<usize>>, // B vecs, each vec is a list of valid patch token ids
    pub segment_ids: Vec<usize>,    // B ids, for mapping onto the palette
}

pub struct Dinov2Batcher {
    index: usize,       // index in the palette
    yield_count: usize, // number of patches the already yielded
    len: Option<usize>,
    token_rel: usize,
    model_rel: usize,
    batch_size: usize,
    mean: Array1<f32>,
    std: Array1<f32>,
    raw_image: Arc<Mutex<RgbImage>>,
    palette: Arc<Mutex<palette::Palette>>,
}

impl Dinov2Batcher {
    pub fn new(
        batch_size: usize,
        raw_image: Arc<Mutex<RgbImage>>,
        palette: Arc<Mutex<palette::Palette>>,
    ) -> Self {
        Self {
            index: 0,
            yield_count: 0,
            len: None,
            token_rel: 14,
            model_rel: 448,
            mean: Array1::from_vec(vec![0.485, 0.456, 0.406]),
            std: Array1::from_vec(vec![0.229, 0.224, 0.225]),
            batch_size,
            raw_image,
            palette,
        }
    }
    pub fn len(&mut self) -> usize {
        // num_patches include all valid and invalid patches. we only need the valid patches
        match self.len {
            Some(len) => len,
            None => {
                let len = self
                    .palette
                    .lock()
                    .unwrap()
                    .valid
                    .iter()
                    .filter(|&&x| x)
                    .count();
                self.len = Some(len);
                len
            }
        }
    }
}

impl Iterator for Dinov2Batcher {
    type Item = Dinov2Batch;
    fn next(&mut self) -> Option<Self::Item> {
        let length = self.len();
        if self.yield_count >= length {
            return None;
        }
        let palette = self.palette.lock().unwrap();
        let raw_image = self.raw_image.lock().unwrap();
        let actual_batch_size = if self.yield_count + self.batch_size > length {
            length - self.yield_count
        } else {
            self.batch_size
        };
        let mut batch =
            Array4::<f32>::uninit((actual_batch_size, self.model_rel, self.model_rel, 3));
        let mut total_token_ids = Vec::new();
        let mut segment_ids = Vec::new();
        let mut count = 0;
        while count < actual_batch_size {
            if !palette.valid[self.index] {
                self.index += 1;
                continue;
            }
            // preprocessing. same as SAM2 but with different size
            let [y, x, size] = palette.bboxes[self.index];
            let patch = raw_image
                .view(x as u32, y as u32, size as u32, size as u32)
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
                .assign_to(batch.slice_mut(s![count, .., .., ..]));
            // get valid patch token ids
            let num_patches = self.model_rel / self.token_rel;
            let stride = size / num_patches;
            let mut token_ids = Vec::new();
            for i in 0..num_patches {
                for j in 0..num_patches {
                    let grid_x_start = x + j * stride;
                    let grid_y_start = y + i * stride;
                    let mut found = false;
                    for grid_y in grid_y_start..grid_y_start + stride {
                        for grid_x in grid_x_start..grid_x_start + stride {
                            if palette.map[(grid_y, grid_x)] == self.index + 1 {
                                token_ids.push(i * num_patches + j);
                                found = true;
                                break;
                            }
                        }
                        if found {
                            break;
                        }
                    }
                }
            }
            total_token_ids.push(token_ids);
            segment_ids.push(self.index + 1);
            self.index += 1;
            count += 1;
        }
        self.yield_count += actual_batch_size;
        unsafe {
            Some(Dinov2Batch {
                image: batch
                    .assume_init()
                    .permuted_axes([0, 3, 1, 2])
                    .as_standard_layout()
                    .to_owned(),
                token_ids: total_token_ids,
                segment_ids: segment_ids.to_owned(),
            })
        }
    }
}

pub struct Dinov2Model {
    model_rel: usize,
    token_rel: usize,
    session: Session,
}

impl Dinov2Model {
    pub fn from_path(path: &str, initialize: bool) -> Result<Self, Box<dyn std::error::Error>> {
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
        Ok(Self {
            model_rel: 448,
            token_rel: 14,
            session,
        })
    }
    pub fn forward(
        &mut self,
        batch_image: Dinov2Batch,
    ) -> Result<Dinov2Output, Box<dyn std::error::Error>> {
        let batch_size = batch_image.image.shape()[0];
        let input_tensor = TensorRef::from_array_view(&batch_image.image)?;
        let output = self.session.run(ort::inputs!["img" => input_tensor])?;
        let patch_tokens = &output["patch_tokens"];
        let patch_tokens = patch_tokens
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?;
        let feature_dim = patch_tokens.shape()[2];
        let mut features = Array2::<f32>::uninit([batch_size, feature_dim]);
        for i in 0..batch_size {
            let mut feature = Array1::<f32>::zeros([feature_dim]);
            for token_id in batch_image.token_ids[i].iter() {
                let token_feature = &patch_tokens.slice(s![i, *token_id, ..]);
                feature = feature + token_feature;
            }
            feature = feature / batch_image.token_ids[i].len() as f32; // global mean pooling
            feature.view().assign_to(features.slice_mut(s![i, ..]));
        }
        unsafe {
            Ok(Dinov2Output {
                segment_ids: batch_image.segment_ids,
                features: features.assume_init().to_owned(),
            })
        }
    }
}
