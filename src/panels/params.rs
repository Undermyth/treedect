use eframe::egui;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;

use crate::panels::canvas::Layer;
use crate::panels::{actions, global};

pub struct ParamsPanel {
    model_path_receiver: Option<Receiver<String>>,
    depth_progress_receiver: Option<Receiver<f32>>,
    depth_receiver: Option<Receiver<Vec<[usize; 2]>>>,
}

impl ParamsPanel {
    pub fn new() -> Self {
        Self {
            model_path_receiver: None,
            depth_progress_receiver: None,
            depth_receiver: None,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        egui::Grid::new("params_grid")
            .num_columns(3)
            .min_col_width(100.0)
            .show(ui, |ui| {
                ui.label("Segment Resolution");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.segment_rel)
                        .speed(20)
                        .suffix(" px"),
                );
                ui.end_row();

                ui.label("Model Path");
                ui.text_edit_singleline(&mut global.params.model_dir);

                // check receiver
                if let Some(receiver) = &self.model_path_receiver {
                    if let Ok(path) = receiver.try_recv() {
                        global.params.model_dir = path;
                    }
                }

                if ui
                    .add_sized([50.0, ui.available_height()], egui::Button::new("Select"))
                    .clicked()
                {
                    let (sender, receiver) = channel();
                    self.model_path_receiver = Some(receiver);
                    actions::select_model_path_action(sender);
                }
                ui.end_row();

                // ui.label("Depth Model");
                // egui::ComboBox::from_id_salt("depth_model")
                //     .width(100.0)
                //     .selected_text(global.params.depth_model_name.as_deref().unwrap_or(""))
                //     .show_ui(ui, |ui| {
                //         ui.selectable_value(
                //             &mut global.params.depth_model_name,
                //             Some("depv2_base".to_string()),
                //             "DEPTH_V2_BASE",
                //         );
                //     });
                // if ui
                //     .add_sized([50.0, ui.available_height()], egui::Button::new(" Load "))
                //     .clicked()
                // {
                //     actions::load_depth_model_action(global);
                // }
                // ui.end_row();

                ui.label("Segment Model");
                egui::ComboBox::from_id_salt("segment_model")
                    .width(100.0)
                    .selected_text(global.params.segment_model_name.as_deref().unwrap_or(""))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut global.params.segment_model_name,
                            Some("sam2_small".to_string()),
                            "SAM2_HIERA_SMALL",
                        );
                        ui.selectable_value(
                            &mut global.params.segment_model_name,
                            Some("sam2_large".to_string()),
                            "SAM2_HIERA_LARGE",
                        );
                    });
                if ui
                    .add_sized([50.0, ui.available_height()], egui::Button::new(" Load "))
                    .clicked()
                {
                    actions::load_segment_model_action(global);
                }
                ui.end_row();

                ui.label("Classification Model");
                egui::ComboBox::from_id_salt("classify_model")
                    .width(100.0)
                    .selected_text(global.params.classify_model_name.as_deref().unwrap_or(""))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut global.params.classify_model_name,
                            Some("dinov2_base".to_string()),
                            "DINOv2_BASE_REG 448x448",
                        );
                    });
                if ui
                    .add_sized([50.0, ui.available_height()], egui::Button::new(" Load "))
                    .clicked()
                {
                    actions::load_classify_model_action(global);
                }
                ui.end_row();

                ui.label("Batch Size");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.batch_size).speed(2),
                );
                ui.end_row();

                ui.label("Luminance Threshold");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.luminance_filt).speed(1),
                );
                ui.end_row();

                ui.label("X Scan Interval");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.x_scan_interval)
                        .speed(1)
                        .suffix(" px"),
                );
                ui.end_row();

                ui.label("Y Scan Interval");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.y_scan_interval)
                        .speed(1)
                        .suffix(" px"),
                );
                ui.end_row();

                ui.label("Merge Threshold");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.merge_thr).speed(0.01),
                );
                ui.end_row();

                ui.label("Grid");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.n_grid).speed(1),
                );
                ui.end_row();

                ui.label("Classes");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.n_classes).speed(1),
                );
                ui.end_row();

                ui.label("Show Cluster IDs");
                ui.checkbox(&mut global.params.show_cluster_ids, "Show");
                ui.end_row();
            });
    }
}
