use crate::panels::canvas;
use crate::worker::GUIChannel;

pub enum ProgressState {
    Loading(String),
    Processing(String, f32),
    Finished,
    Error(String),
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
    pub segment_rel: i32,
    /// directory path where ONNX models are stored.
    pub model_dir: String,
    /// model used for segmentation. Selected from SAM2 family.
    pub segment_model_name: Option<String>,
    /// model used for classification. Selected from DINOv2 family.
    pub classify_model_name: Option<String>,
    /// whether to use height mapping for sampling points estimation.
    pub use_height_sampling: bool,
    /// if using grid sampling, the interval between adjacent sampling points. (in pixels)
    pub grid_sampling_interval: usize,
    /// batch size for both segmentation and classification.
    pub batch_size: usize,
}

impl Params {
    pub fn new() -> Self {
        Self {
            segment_rel: 1024,
            model_dir: "./output_models".to_string(),
            segment_model_name: None,
            classify_model_name: None,
            use_height_sampling: false,
            grid_sampling_interval: 320,
            batch_size: 8,
        }
    }
}

pub struct GlobalState {
    pub layers: Vec<canvas::Layer>,
    pub progress_state: ProgressState,
    pub canvas_state: canvas::CanvasState,
    pub gui_channel: GUIChannel,
    pub params: Params,
    pub raw_image: Option<canvas::LayerImage>,
    pub sampling_points: Option<Vec<[usize; 2]>>,
    pub select_pos: [usize; 2],
}

impl GlobalState {
    pub fn new(gui_channel: GUIChannel) -> Self {
        Self {
            layers: Vec::<canvas::Layer>::new(),
            progress_state: ProgressState::Finished,
            canvas_state: canvas::CanvasState::default(),
            gui_channel: gui_channel,
            params: Params::new(),
            raw_image: None,
            sampling_points: None,
            select_pos: [0, 0],
        }
    }
}
