// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use eframe::egui::Visuals;
use eframe::epaint::text::{FontInsert, InsertFontFamily};
use std::env;
use std::path::PathBuf;

mod models;
mod panels;
mod utils;

/// 初始化 Python 环境变量，必须在任何 Python 操作之前调用
fn init_python_environment() {
    // 获取可执行文件所在目录
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let python_home = exe_dir.join("python");

            // 检查 Python 环境是否存在
            if python_home.exists() {
                // 设置 PYTHONHOME - 这是 Python 找到标准库的关键
                unsafe {
                    env::set_var("PYTHONHOME", &python_home);
                }

                // 设置 PYTHONPATH - 包含 Lib 和 DLLs
                let lib_path = python_home.join("Lib");
                let dlls_path = python_home.join("DLLs");
                let python_path = format!(
                    "{};{}",
                    lib_path.to_string_lossy(),
                    dlls_path.to_string_lossy()
                );
                unsafe {
                    env::set_var("PYTHONPATH", python_path);
                }

                // 设置 PYTHONNOUSERSITE=1 避免加载用户 site-packages
                unsafe {
                    env::set_var("PYTHONNOUSERSITE", "1");
                }

                // 禁用 PYTHONDONTWRITEBYTECODE 避免生成 .pyc 文件
                unsafe {
                    env::set_var("PYTHONDONTWRITEBYTECODE", "1");
                }

                eprintln!("[PythonEnv] PYTHONHOME set to: {}", python_home.display());
            } else {
                eprintln!(
                    "[PythonEnv] Warning: Python environment not found at: {}",
                    python_home.display()
                );
            }
        }
    }
}

fn main() -> eframe::Result {
    // 必须在最开始就初始化 Python 环境变量
    // 这样 PyO3 初始化 Python 时就能正确找到标准库
    init_python_environment();

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
