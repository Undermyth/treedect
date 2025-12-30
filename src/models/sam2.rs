use ort::execution_providers::WebGPUExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

pub struct SAM2Model {
    pub encoder_session: Session,
    pub decoder_session: Session,
}

impl SAM2Model {
    pub fn from_path(
        encoder_path: &str,
        decoder_path: &str,
        initialize: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if initialize {
            log::info!("Initializing ONNX Runtime with WebGPU execution provider");
            ort::init()
                .with_execution_providers([WebGPUExecutionProvider::default()
                    .build()
                    .error_on_failure()])
                .commit()?;
        }
        log::info!("Loading encoder model from: {}", encoder_path);
        let encoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;
        log::info!("Loading decoder model from: {}", decoder_path);
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;
        log::info!("Successfully loaded both encoder and decoder models");
        Ok(Self {
            encoder_session,
            decoder_session,
        })
    }
}
