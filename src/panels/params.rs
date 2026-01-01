use eframe::egui;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;

use crate::panels::canvas::Layer;
use crate::panels::{actions, global};

pub struct ParamsPanel {
    model_path_receiver: Option<Receiver<String>>,
}

impl ParamsPanel {
    pub fn new() -> Self {
        Self {
            model_path_receiver: None,
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
                        .speed(100)
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
                    .add_sized([30.0, ui.available_height()], egui::Button::new("Select"))
                    .clicked()
                {
                    let (sender, receiver) = channel();
                    self.model_path_receiver = Some(receiver);
                    actions::select_model_path_action(sender);
                }
                ui.end_row();

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
                    });
                if ui
                    .add_sized([30.0, ui.available_height()], egui::Button::new(" Load "))
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
                    .add_sized([30.0, ui.available_height()], egui::Button::new(" Load "))
                    .clicked()
                {
                    actions::load_classify_model_action(global);
                }
                ui.end_row();

                ui.label("H-Aware Sampling");
                ui.checkbox(&mut global.params.use_height_sampling, "Enable");
                if ui
                    .add_sized([30.0, ui.available_height()], egui::Button::new(" Start "))
                    .clicked()
                {
                    if global.layers.len() == 0 {
                        global.progress_state = global::ProgressState::Error(
                            "No image loaded. Please load an image first.".to_string(),
                        );
                        return;
                    }
                    // pop out the previous generated sampling points layer
                    if global.layers.len() == 2 {
                        global.layers.pop();
                    }
                    if global.layers.len() > 2 {
                        global.progress_state = global::ProgressState::Error(
                            "Sampling should be done before segmentation & classification. \
                             Please reload the image to start a new sampling."
                                .to_string(),
                        );
                        return;
                    }
                    if !global.params.use_height_sampling {
                        let [width, height] = global.layers[0].get_image_size();
                        let start_time = std::time::Instant::now();
                        let mut sampling_points = actions::grid_sampling_action(
                            global.params.grid_sampling_interval,
                            width,
                            height,
                        );
                        let grid_time = start_time.elapsed();
                        log::info!("Grid sampling took: {:?}", grid_time);

                        let start_time = std::time::Instant::now();
                        let sampling_points = actions::filter_sampling_action(
                            &mut sampling_points,
                            global.raw_image.as_ref().unwrap(),
                        );
                        let filter_time = start_time.elapsed();
                        log::info!("Filter sampling took: {:?}", filter_time);
                        log::info!("Number of sampling points: {}", sampling_points.len());
                        global.layers.push(Layer::from_sampling_points(
                            &sampling_points,
                            width,
                            height,
                            global.params.segment_rel as usize,
                        ));
                        global.sampling_points = Some(sampling_points);
                    }
                }
                ui.end_row();

                ui.label("Sampling Interval");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.grid_sampling_interval)
                        .speed(5)
                        .suffix(" px"),
                );
                ui.end_row();

                ui.label("Batch Size");
                ui.add_sized(
                    ui.available_size_before_wrap(),
                    egui::DragValue::new(&mut global.params.batch_size).speed(2),
                );
                ui.end_row();
            });
    }
}
