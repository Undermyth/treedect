// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use eframe::egui::Visuals;

mod models;
mod panels;

fn main() -> eframe::Result {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let detail_logging = args.iter().any(|arg| arg == "--detail-logging");

    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    log::info!("Treedect by egui");
    log::info!("ONNX Runtime version: {}", ort::MINOR_VERSION);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "treedect(beta)",
        options,
        Box::new(|cc| Ok(Box::new(TreeDectApp::new(cc, detail_logging)))),
    )
}

struct TreeDectApp {
    global: panels::global::GlobalState,
    left_panel: panels::left::LeftPanel,
    right_panel: panels::right::RightPanel,
    central_panel: panels::central::CentralPanel,
}

impl TreeDectApp {
    fn new(_cc: &eframe::CreationContext, detail_logging: bool) -> Self {
        // layers will be stacked with the order 0, 1, ...
        Self {
            global: panels::global::GlobalState::new(detail_logging),
            left_panel: panels::left::LeftPanel::new(),
            right_panel: panels::right::RightPanel::new(),
            central_panel: panels::central::CentralPanel::new(),
        }
    }
}

impl eframe::App for TreeDectApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(Visuals::light());
        egui::SidePanel::left("layer_control").show(ctx, |ui| {
            self.left_panel.ui(ui, &mut self.global);
        });

        egui::SidePanel::right("control_and_output")
            .exact_width(300.0)
            .show(ctx, |ui| {
                self.right_panel.ui(ui, &mut self.global);
            });

        egui::CentralPanel::default()
            .show(ctx, |ui| self.central_panel.ui(ctx, ui, &mut self.global));
    }
}
