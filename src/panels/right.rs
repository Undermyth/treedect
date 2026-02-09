use eframe::egui;
use std::sync::mpsc::{Receiver, Sender};

use crate::panels::actions;
use crate::panels::global;
use crate::panels::params;
use crate::panels::table;

pub struct RightPanel {
    params_panel: params::ParamsPanel,
    table_panel: table::TablePanel,
    save_path_receiver: Option<Receiver<String>>,
}

impl RightPanel {
    pub fn new() -> Self {
        Self {
            params_panel: params::ParamsPanel::new(),
            table_panel: table::TablePanel::new(),
            save_path_receiver: None,
        }
    }

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
                                // ui.set_width(ui.available_width());
                                self.params_panel.ui(ui, global);
                                // for i in 0..32 {
                                //     ui.label(format!("Control item {}", i));
                                // }
                            });
                    },
                );

                ui.separator();

                if let Some(receiver) = &self.save_path_receiver {
                    if let Ok(path) = receiver.try_recv() {
                        let result = global.score_table.as_mut().unwrap().export_to_csv(path);
                        if let Err(e) = result {
                            global.progress_state = global::ProgressState::Error(e.to_string());
                        }
                        self.save_path_receiver = None;
                    }
                }

                // Lower part, Output Panel - 50% height
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), panel_height * 0.5),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let sort_text = if global.sorted {
                            "Sort ↑"
                        } else {
                            "Sort ↓"
                        };
                        ui.horizontal(|ui| {
                            if ui.button(sort_text).clicked() {
                                if global.score_table.is_none() {
                                    return;
                                }
                                if !global.sorted {
                                    global.score_table.as_mut().unwrap().sort_by_score(true);
                                    global.sorted = true;
                                } else {
                                    global.score_table.as_mut().unwrap().sort_by_score(false);
                                    global.sorted = false;
                                }
                            }
                            if ui.button("Sort By ID").clicked() {
                                if global.score_table.is_none() {
                                    return;
                                }
                                global.score_table.as_mut().unwrap().sort_by_id(false);
                            }
                            if ui.button("Export..").clicked() {
                                let (sender, receiver) = std::sync::mpsc::channel();
                                self.save_path_receiver = Some(receiver);
                                actions::save_csv_action(sender);
                            }
                        });
                        ui.add_space(10.0); // Add some space between buttons and table
                        self.table_panel.ui(ui, global);
                    },
                );
            },
        );
    }
}
