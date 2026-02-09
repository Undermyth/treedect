// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use eframe::egui::Visuals;
use eframe::epaint::text::{FontInsert, InsertFontFamily};

mod models;
mod panels;
mod utils;

fn main() -> eframe::Result {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    log::info!("Command line arguments: {:?}", args);
    let detail_logging = args.iter().any(|arg| arg == "--detail-logging");
    if detail_logging {
        log::info!("Detailed Logging is enabled");
    }

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

// Demonstrates how to add a font to the existing ones
fn add_font(ctx: &egui::Context) {
    ctx.add_font(FontInsert::new(
        "my_font",
        egui::FontData::from_static(include_bytes!("./IosevkaTermNerdFont-Regular.ttf")),
        vec![
            InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: egui::epaint::text::FontPriority::Highest,
            },
            InsertFontFamily {
                family: egui::FontFamily::Monospace,
                priority: egui::epaint::text::FontPriority::Lowest,
            },
        ],
    ));
}

// Demonstrates how to replace all fonts.
fn replace_fonts(ctx: &egui::Context) {
    // Start with the default fonts (we will be adding to them rather than replacing them).
    let mut fonts = egui::FontDefinitions::default();

    // Install my own font (maybe supporting non-latin characters).
    // .ttf and .otf files supported.
    fonts.font_data.insert(
        "my_font".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "./IosevkaTermNerdFont-Regular.ttf"
        ))),
    );

    // Put my font first (highest priority) for proportional text:
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "my_font".to_owned());

    // Put my font as last fallback for monospace:
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("my_font".to_owned());

    // Tell egui to use these fonts:
    ctx.set_fonts(fonts);
}

struct TreeDectApp {
    global: panels::global::GlobalState,
    left_panel: panels::left::LeftPanel,
    right_panel: panels::right::RightPanel,
    central_panel: panels::central::CentralPanel,
}

impl TreeDectApp {
    fn new(cc: &eframe::CreationContext, detail_logging: bool) -> Self {
        replace_fonts(&cc.egui_ctx);
        add_font(&cc.egui_ctx);
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
            .exact_width(320.0)
            .show(ctx, |ui| {
                self.right_panel.ui(ui, &mut self.global);
            });

        egui::CentralPanel::default()
            .show(ctx, |ui| self.central_panel.ui(ctx, ui, &mut self.global));
    }
}
