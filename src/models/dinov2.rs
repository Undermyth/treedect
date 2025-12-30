use ort::execution_providers::WebGPUExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

pub struct Dinov2Model {
    session: Session,
}

impl Dinov2Model {
    pub fn from_path(path: &str, initialize: bool) -> Result<Self, Box<dyn std::error::Error>> {
        if initialize {
            log::info!("Initializing ONNX Runtime with WebGPU execution provider");
            ort::init()
                .with_execution_providers([WebGPUExecutionProvider::default()
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
        Ok(Self { session })
    }
}
