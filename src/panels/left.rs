use eframe::egui;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;

use crate::panels::actions;
use crate::panels::canvas;
use crate::panels::components;
use crate::panels::global;

pub struct ActionPanel {
    image_select_receiver: Option<Receiver<String>>,
    image_load_receiver: Option<Receiver<Result<canvas::Layer, String>>>,
}

impl ActionPanel {
    pub fn new() -> Self {
        Self {
            image_select_receiver: None,
            image_load_receiver: None,
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
                            global.layers.clear();
                            global.layers.push(layer);
                            global.progress_state = global::ProgressState::Finished;
                        }
                        Err(e) => {
                            eprintln!("Failed to load image: {}", e);
                            global.progress_state = global::ProgressState::Finished;
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
                        // 使用tokio运行时执行异步任务
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async move {
                            match canvas::Layer::from_path(name, path).await {
                                Ok(layer) => {
                                    let _ = async_sender.send(Ok(layer));
                                }
                                Err(e) => {
                                    let _ = async_sender.send(Err(e.to_string()));
                                }
                            }
                            ctx.request_repaint();
                        });
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
            if ui
                .add(components::wide_button(
                    "Segmentation",
                    ui.available_width(),
                ))
                .clicked()
            {}
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
