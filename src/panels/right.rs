use eframe::egui;

use crate::panels::global;

pub struct RightPanel {}

impl RightPanel {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(unused_variables)]
    pub fn ui(&mut self, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        ui.allocate_ui_with_layout(
            ui.available_size(),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let panel_height = ui.available_height();

                // Upper part, Control Parameter Panel - 50% height
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), panel_height * 0.5),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.heading("Control Panel");
                        ui.add_space(10.0);
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                for i in 0..32 {
                                    ui.label(format!("Control item {}", i));
                                }
                            });
                    },
                );

                ui.separator();

                // Lower part, Output Panel - 50% height
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), panel_height * 0.5),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.heading("Output");
                        // 这里可以添加输出信息
                    },
                );
            },
        );
    }
}
