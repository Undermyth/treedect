// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use eframe::egui::Visuals;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcOneShotServer;

use crate::worker::GUIChannel;

mod models;
mod panels;
mod worker;

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();

    if args.len() <= 1 {
        // No additional command line arguments, enter egui process
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
        log::info!("Treedect by egui");
        log::info!("ONNX Runtime version: {}", ort::MINOR_VERSION);

        // Create a one-shot server to bootstrap the connection
        let (server, server_name) = IpcOneShotServer::<GUIChannel>::new().unwrap();

        // launch worker process
        let worker_handler = std::process::Command::new(&args[0])
            .arg("--worker")
            .arg(server_name)
            .spawn()
            .expect("Failed to launch worker process");

        // Accept the connection from the worker
        let (_, gui_channel) = server.accept().unwrap();

        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 800.0]),
            ..Default::default()
        };
        eframe::run_native(
            "treedect(beta)",
            options,
            Box::new(|cc| Ok(Box::new(TreeDectApp::new(cc, gui_channel, worker_handler)))),
        )
    } else {
        // Additional command line arguments exist
        // First argument should be "--worker", followed by server name
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
        if args.len() >= 3 && args[1] == "--worker" {
            let server_name = &args[2];
            log::info!("Worker mode with server name: {}", server_name);

            worker::main(server_name);

            // Actual worker logic would go here
        } else {
            eprintln!("Invalid command line arguments");
            std::process::exit(1);
        }

        Ok(())
    }
}

struct TreeDectApp {
    global: panels::global::GlobalState,
    left_panel: panels::left::LeftPanel,
    right_panel: panels::right::RightPanel,
    central_panel: panels::central::CentralPanel,
    worker_handler: std::process::Child,
}

impl TreeDectApp {
    fn new(
        _cc: &eframe::CreationContext,
        gui_channel: worker::GUIChannel,
        worker_handler: std::process::Child,
    ) -> Self {
        // layers will be stacked with the order 0, 1, ...
        Self {
            global: panels::global::GlobalState::new(gui_channel),
            left_panel: panels::left::LeftPanel::new(),
            right_panel: panels::right::RightPanel::new(),
            central_panel: panels::central::CentralPanel::new(),
            worker_handler,
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

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        log::info!("Shutting down worker process");
        let _ = self.worker_handler.kill();
        let _ = self.worker_handler.wait();
    }
}
