use eframe::egui;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};

use crate::panels::actions;
use crate::panels::canvas;
use crate::panels::canvas::Layer;
use crate::panels::canvas::LayerImage;
use crate::panels::components;
use crate::panels::global;
use crate::panels::palette;

pub struct ActionPanel {
    image_select_receiver: Option<Receiver<String>>,
    image_load_receiver: Option<Receiver<Result<canvas::Layer, String>>>,
    segment_progress_receiver: Option<Receiver<f32>>,
    segment_receiver: Option<Receiver<palette::Palette>>,
    classify_progress_receiver: Option<Receiver<f32>>,
    classify_receiver: Option<Receiver<bool>>,
    image_save_receiver: Option<Receiver<String>>,
}

impl ActionPanel {
    pub fn new() -> Self {
        Self {
            image_select_receiver: None,
            image_load_receiver: None,
            segment_progress_receiver: None,
            segment_receiver: None,
            classify_progress_receiver: None,
            classify_receiver: None,
            image_save_receiver: None,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        ui.heading("Actions");
        ui.add_space(10.0);
        ui.vertical_centered_justified(|ui| {
            // 检查是否有异步加载完成的图像
            if let Some(receiver) = &self.image_load_receiver {
                if let Ok(result) = receiver.try_recv() {
                    match result {
                        Ok(layer) => {
                            if let LayerImage::RGBImage(image) = layer.raw_image.clone().unwrap() {
                                global.raw_image = Some(Arc::new(Mutex::new(image)));
                            }
                            global.layers.clear();
                            global.layers.push(layer);
                            global.progress_state =
                                global::ProgressState::Finished("Image loaded".to_string());
                        }
                        Err(e) => {
                            eprintln!("Failed to load image: {}", e);
                            global.progress_state = global::ProgressState::Error(e.to_string());
                        }
                    }
                    // 清除接收器
                    self.image_load_receiver = None;
                }
            }

            // 检查是否有通过同步通道传入的图像路径
            if let Some(receiver) = &self.image_select_receiver {
                if let Ok(image_path) = receiver.try_recv() {
                    // 启动异步图像加载
                    global.progress_state =
                        global::ProgressState::Loading("Loading image...".to_string());

                    // 创建异步通道
                    let (async_sender, async_receiver) = channel();
                    self.image_load_receiver = Some(async_receiver);

                    // 克隆需要的变量
                    let name = "Image Layer".to_string();
                    let path = image_path.clone();
                    let ctx = ui.ctx().clone();

                    // 在后台线程中执行异步加载
                    std::thread::spawn(move || {
                        match canvas::Layer::from_path(name, path) {
                            Ok(layer) => {
                                let _ = async_sender.send(Ok(layer));
                            }
                            Err(e) => {
                                let _ = async_sender.send(Err(e.to_string()));
                            }
                        }
                        ctx.request_repaint();
                    });
                }
            }

            // 按钮本体和触发逻辑
            if ui
                .add(components::wide_button("Load Image", ui.available_width()))
                .clicked()
            {
                let (async_sender, async_receiver) = channel();
                self.image_select_receiver = Some(async_receiver);
                actions::load_image_action(ui.ctx().clone(), async_sender);
            }
        });

        ui.vertical_centered_justified(|ui| {
            if let Some(receiver) = &self.segment_progress_receiver {
                if let Ok(progress) = receiver.try_recv() {
                    global.progress_state = global::ProgressState::Processing(
                        "Running Segmentation".to_string(),
                        progress,
                    );
                    ui.ctx().request_repaint();
                }
            }
            if let Some(receiver) = &self.segment_receiver {
                if let Ok(palette) = receiver.try_recv() {
                    global.layers.push(canvas::Layer::from_palette(
                        "Segmentation".to_string(),
                        &palette,
                    ));
                    global.palette = Some(Arc::new(Mutex::new(palette)));
                    global.progress_state =
                        global::ProgressState::Finished("Segmentation finished".to_string());
                    ui.ctx().request_repaint();
                    self.segment_progress_receiver = None;
                    self.segment_receiver = None;
                }
            }

            if ui
                .add(components::wide_button(
                    "Segmentation",
                    ui.available_width(),
                ))
                .clicked()
            {
                if global.raw_image.is_none() {
                    global.progress_state =
                        global::ProgressState::Error("No image loaded".to_string());
                    return;
                };
                if global.segment_model.is_none() {
                    global.progress_state =
                        global::ProgressState::Error("No segmentation model loaded".to_string());
                    return;
                }
                if global.sampling_points.is_none() {
                    global.progress_state =
                        global::ProgressState::Error("No sampling points generated".to_string());
                    return;
                }
                let (progress_sender, progress_receiver) = channel();
                let (segment_sender, segment_receiver) = channel();
                self.segment_progress_receiver = Some(progress_receiver);
                self.segment_receiver = Some(segment_receiver);
                actions::segment_action(global, progress_sender, segment_sender);
            }
        });

        ui.vertical_centered_justified(|ui| {
            if let Some(receiver) = &self.classify_progress_receiver {
                if let Ok(progress) = receiver.try_recv() {
                    global.progress_state = global::ProgressState::Processing(
                        "Running Classification".to_string(),
                        progress,
                    );
                    ui.ctx().request_repaint();
                }
            }
            if let Some(receiver) = &self.classify_receiver {
                if let Ok(_finished) = receiver.try_recv() {
                    let palette = global.palette.as_ref().unwrap();
                    let palette = palette.clone();
                    global.layers.push(Layer::from_palette_cluster(
                        "Classification".to_string(),
                        palette,
                    ));
                    global.layers[2].visible = false;
                    actions::get_importance_score(global);
                    ui.ctx().request_repaint();
                    global.progress_state =
                        global::ProgressState::Finished("Classification finished".to_string());
                    self.classify_progress_receiver = None;
                    self.classify_receiver = None;
                }
            }
            if ui
                .add(components::wide_button(
                    "Classification",
                    ui.available_width(),
                ))
                .clicked()
            {
                if global.raw_image.is_none() {
                    global.progress_state =
                        global::ProgressState::Error("No image loaded".to_string());
                    return;
                }
                if global.classify_model.is_none() {
                    global.progress_state =
                        global::ProgressState::Error("No classification model loaded".to_string());
                    return;
                }
                if global.layers.len() < 3 {
                    global.progress_state =
                        global::ProgressState::Error("Segmentation is not executed".to_string());
                    return;
                }
                let (progress_sender, progress_receiver) = channel();
                let (classify_sender, classify_receiver) = channel();
                self.classify_progress_receiver = Some(progress_receiver);
                self.classify_receiver = Some(classify_receiver);
                actions::classify_action(global, progress_sender, classify_sender);
            }
        });

        ui.vertical_centered_justified(|ui| {
            if let Some(receiver) = &self.image_save_receiver {
                if let Ok(path) = receiver.try_recv() {
                    actions::export_image_action(global, path);
                    self.image_save_receiver = None;
                }
            }
            if ui
                .add(components::wide_button(
                    "Export Image...",
                    ui.available_width(),
                ))
                .clicked()
            {
                let (sender, receiver) = channel();
                self.image_save_receiver = Some(receiver);
                actions::save_img_action(sender);
            }
        });
    }
}

pub struct LeftPanel {
    action_panel: ActionPanel, // lower part layer panel is not stateful. integrated.
}

impl LeftPanel {
    pub fn new() -> Self {
        Self {
            action_panel: ActionPanel::new(),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        ui.allocate_ui_with_layout(
            ui.available_size(),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let panel_height = ui.available_height();

                // Upper part, Action Center - 50% height
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), panel_height * 0.5),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        self.action_panel.ui(ui, global);
                    },
                );

                ui.separator();

                // Lower part, Layer List - 50% height
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), panel_height * 0.5),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.heading("Layers");
                        ui.add_space(10.0);
                        // 倒序遍历以模拟 Photoshop 的图层列表（上面的是顶层）
                        for (i, layer) in global.layers.iter_mut().enumerate().rev() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut layer.visible, "");
                                    ui.label(format!("Layer {}: {}", i, layer.name));
                                });
                                ui.add(
                                    egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                        .text("Opacity"),
                                );
                            });
                        }
                    },
                );
            },
        );
    }
}
