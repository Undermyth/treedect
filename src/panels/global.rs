use std::sync::{Arc, Mutex};

use crate::models::dam2::DAM2Model;
use crate::models::dinov2::Dinov2Model;
use crate::models::sam2::SAM2Model;
use crate::panels::canvas;
use crate::panels::palette;
use crate::utils::score::Table;

pub enum ProgressState {
    Loading(String),
    Processing(String, f32),
    Finished(String),
    Error(String),
}

pub enum LayerType {
    Image,
    Segmentation,
    Classification,
}

/// This struct contains most of the parameters to control segmentation and classification.
pub struct Params {
    /// segmentation resolution, in pixels. default to 512.
    ///
    /// This parameter is used for high resolution processing, since SAM2 only supports 1024x1024.
    /// Patches of size rel x rel will be truncated around each sampling point, and feed into SAM2.
    ///
    /// Larger resolution will make the picture clearer; However, the global noise will also increase.
    /// Note that rel >= 1024 makes no sense since the intrisic resolution of SAM2 is 1024x1024.
    /// Even larger resolution will cause downsampling inside the model.
    ///
    /// Smaller resolution will have better resolution, but with the assumption that
    /// **the recognized object should be small enough to fall into the patch**.
    ///
    /// This resolution is also used for depth estimation as the sliding window size.
    /// The instrinsic resolution of Depth Anything V2 is 518x518, so downsampling is almost always needed.
    pub segment_rel: i32,
    /// directory path where ONNX models are stored.
    pub model_dir: String,
    // /// model used for depth estimation. Selected from Depth Anything V2 family.
    // pub depth_model_name: Option<String>,
    /// model used for segmentation. Selected from SAM2 family.
    pub segment_model_name: Option<String>,
    /// model used for classification. Selected from DINOv2 family.
    pub classify_model_name: Option<String>,
    // /// whether to use height mapping for sampling points estimation.
    // pub use_height_sampling: bool,
    // /// if using grid sampling, the interval between adjacent sampling points. (in pixels)
    // pub grid_sampling_interval: usize,
    /// batch size for both segmentation and classification. (should be <= 4 as limited by ONNX bugs)
    pub batch_size: usize,
    // /// radius for local maximum dilation. larger value will result in sparser sampling points.
    // pub dilation_radius: usize,
    // /// radius for NMS. larger value will result in sparser sampling points.
    // pub nms_radius: usize,
    /// split the image into n_grid * n_grid to calculate the importance score.
    pub n_grid: usize,
    /// number of classes for clustering.
    pub n_classes: usize,
    /// whether to display the cluster ids.
    pub show_cluster_ids: bool,
    /// luminance threshold for black pixels.
    pub luminance_filt: u8,
    /// scanning interval in X direction.
    pub x_scan_interval: usize,
    /// scanning interval in Y direction.
    pub y_scan_interval: usize,
    /// overlap ratio for merging segments.
    pub merge_thr: f32,
}

impl Params {
    pub fn new() -> Self {
        Self {
            segment_rel: 1024,
            model_dir: "./output_models".to_string(),
            // depth_model_name: None,
            segment_model_name: None,
            classify_model_name: None,
            // use_height_sampling: false,
            // grid_sampling_interval: 320,
            batch_size: 4,
            // dilation_radius: 70,
            // nms_radius: 150,
            n_grid: 4,
            n_classes: 5,
            show_cluster_ids: true,
            luminance_filt: 80,
            x_scan_interval: 50,
            y_scan_interval: 60,
            merge_thr: 0.8,
        }
    }
}

pub struct GlobalState {
    pub layers: Vec<canvas::Layer>,
    pub progress_state: ProgressState,
    pub canvas_state: canvas::CanvasState,
    pub params: Params,
    pub detail_logging: bool,
    pub is_cpu: bool,
    pub depth_model: Option<Arc<Mutex<DAM2Model>>>,
    pub segment_model: Option<Arc<Mutex<SAM2Model>>>,
    pub classify_model: Option<Arc<Mutex<Dinov2Model>>>,
    pub raw_image: Option<Arc<Mutex<image::RgbImage>>>,
    pub sampling_points: Option<Vec<[usize; 2]>>,
    pub palette: Option<Arc<Mutex<palette::Palette>>>,
    pub select_pos: [usize; 2],
    pub score_table: Option<Table>,
    pub sorted: bool,
}

impl GlobalState {
    pub fn new(detail_logging: bool, is_cpu: bool) -> Self {
        Self {
            layers: Vec::<canvas::Layer>::new(),
            canvas_state: canvas::CanvasState::default(),
            progress_state: ProgressState::Finished("All processing finished".to_string()),
            params: Params::new(),
            detail_logging: detail_logging,
            is_cpu: is_cpu,
            depth_model: None,
            segment_model: None,
            classify_model: None,
            raw_image: None,
            sampling_points: None,
            palette: None,
            select_pos: [0, 0],
            score_table: None,
            sorted: false,
        }
    }
    pub fn get_layer(&mut self, layer_type: LayerType) -> &mut canvas::Layer {
        match layer_type {
            LayerType::Image => {
                assert!(self.layers.len() > 0);
                &mut self.layers[0]
            }
            LayerType::Segmentation => {
                assert!(self.layers.len() > 1);
                &mut self.layers[1]
            }
            LayerType::Classification => {
                assert!(self.layers.len() > 2);
                &mut self.layers[2]
            }
        }
    }
}
