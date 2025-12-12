use crate::panels::canvas;

pub enum ProgressState {
    Loading(String),
    Processing(String, f32),
    Finished,
}

pub struct GlobalState {
    pub layers: Vec<canvas::Layer>,
    pub progress_state: ProgressState,
    pub canvas_state: canvas::CanvasState,
}

impl GlobalState {
    pub fn new() -> Self {
        Self {
            layers: Vec::<canvas::Layer>::new(),
            canvas_state: canvas::CanvasState::default(),
            progress_state: ProgressState::Finished,
        }
    }
}
