use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

pub struct Dinov2Model {
    session: Session,
}

impl Dinov2Model {
    pub fn from_path(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Loading model from: {}", path);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;
        log::info!("Successfully loaded model");
        Ok(Self { session })
    }
}
