#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{egui, epaint};
use std::sync::mpsc::{Receiver, Sender, channel};

mod actions;
mod canvas;
mod components;

fn main() -> eframe::Result {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Egui Layers Example",
        options,
        Box::new(|cc| Ok(Box::new(TreeDectApp::new(cc)))),
    )
}

enum ProgressState {
    Loading(String),
    Processing(String, f32),
    Finished,
}

struct TreeDectApp {
    layers: Vec<canvas::Layer>,
    canvas_state: canvas::CanvasState,
    progress_state: ProgressState,
    image_select_channel: (Sender<String>, Receiver<String>),
    // 异步图像加载的结果通道
    image_load_receiver: Option<Receiver<Result<canvas::Layer, String>>>,
}

impl TreeDectApp {
    fn new(_cc: &eframe::CreationContext) -> Self {
        // layers will be stacked with the order 0, 1, ...
        Self {
            layers: Vec::<canvas::Layer>::new(),
            canvas_state: canvas::CanvasState::default(),
            progress_state: ProgressState::Finished,
            image_select_channel: channel(),
            image_load_receiver: None,
        }
    }
}

impl eframe::App for TreeDectApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("layer_control").show(ctx, |ui| {
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
                            ui.heading("Actions");
                            ui.add_space(10.0);
                            ui.vertical_centered_justified(|ui| {
                                // 检查是否有异步加载完成的图像
                                if let Some(receiver) = &self.image_load_receiver {
                                    if let Ok(result) = receiver.try_recv() {
                                        match result {
                                            Ok(layer) => {
                                                self.layers.push(layer);
                                                self.progress_state = ProgressState::Finished;
                                            }
                                            Err(e) => {
                                                eprintln!("Failed to load image: {}", e);
                                                self.progress_state = ProgressState::Finished;
                                            }
                                        }
                                        // 清除接收器
                                        self.image_load_receiver = None;
                                    }
                                }

                                // 检查是否有通过同步通道传入的图像路径
                                if let Ok(image_path) = self.image_select_channel.1.try_recv() {
                                    // 启动异步图像加载
                                    self.progress_state =
                                        ProgressState::Loading("Loading image...".to_string());

                                    // 创建异步通道
                                    let (async_sender, async_receiver) = channel();
                                    self.image_load_receiver = Some(async_receiver);

                                    // 克隆需要的变量
                                    let name = "New Layer".to_string();
                                    let path = image_path.clone();
                                    let ctx = ui.ctx().clone();

                                    // 在后台线程中执行异步加载
                                    std::thread::spawn(move || {
                                        // 使用tokio运行时执行异步任务
                                        let rt = tokio::runtime::Runtime::new().unwrap();
                                        rt.block_on(async move {
                                            match canvas::Layer::from_path_async(name, path).await {
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
                                if ui
                                    .add(components::wide_button(
                                        "Load Image",
                                        ui.available_width(),
                                    ))
                                    .clicked()
                                {
                                    actions::load_image_action(
                                        ui.ctx().clone(),
                                        self.image_select_channel.0.clone(),
                                    );
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
                            })
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
                            for (i, layer) in self.layers.iter_mut().enumerate().rev() {
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
        });

        egui::SidePanel::right("control_and_output")
            .exact_width(300.0)
            .show(ctx, |ui| {
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
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Canvas Area");

            ui.allocate_ui_with_layout(
                ui.available_size(),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    let panel_height = ui.available_height();

                    // Upper part, Canvas - 90% height
                    ui.allocate_ui_with_layout(
                        egui::Vec2::new(ui.available_width(), panel_height * 0.95),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            // 1. 划定画布区域
                            // 自适应填满整个中央面板区域
                            let canvas_size = ui.available_size();
                            let (response, painter) =
                                ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());

                            // 绘制背景框，方便看清画布边界
                            painter.rect_stroke(
                                response.rect,
                                0.0,
                                (1.0, egui::Color32::WHITE),
                                epaint::StrokeKind::Inside,
                            );

                            if self.layers.len() > 0 {
                                let image_size = self.layers[0].get_image_size();
                                canvas::update_drag_and_zoom(
                                    ui,
                                    &response,
                                    &mut self.canvas_state,
                                    canvas_size,
                                    image_size,
                                );
                            }

                            // 2. 循环绘制图层
                            for layer in &mut self.layers {
                                if !layer.visible || layer.opacity <= 0.0 {
                                    continue;
                                }

                                let tex_id = layer.texture_id(ctx);

                                // 计算透明度颜色混合
                                // egui 的 tint 颜色会与纹理像素相乘。
                                // 如果要控制整体透明度，使用带有 Alpha 的白色即可。
                                let tint =
                                    egui::Color32::from_white_alpha((layer.opacity * 255.0) as u8);

                                // 3. 执行绘制，应用缩放和平移变换
                                let image_size = layer.get_image_size();
                                let scaled_size =
                                    egui::vec2(image_size[0] as f32, image_size[1] as f32)
                                        * self.canvas_state.scale;

                                // 图片左上角在屏幕上的位置 = 视口左上角 + 当前偏移
                                let image_min = response.rect.min + self.canvas_state.offset;
                                let image_rect = egui::Rect::from_min_size(image_min, scaled_size);

                                // 设置裁剪区域，这样超出屏幕的部分不会绘制到其他UI上
                                let clipped_painter = painter.with_clip_rect(response.rect);

                                // 绘制图片
                                clipped_painter.image(
                                    tex_id,
                                    image_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ), // UV 坐标
                                    tint,
                                );
                            }
                        },
                    );

                    ui.separator();

                    // Lower part, Progress - 10% height
                    ui.allocate_ui_with_layout(
                        egui::Vec2::new(ui.available_width(), panel_height * 0.05),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            // ui.heading("Progress");
                            // ui.add_space(10.0);
                            ui.vertical_centered_justified(|ui| match &self.progress_state {
                                ProgressState::Loading(text) => ui.horizontal(|ui| {
                                    ui.label(text);
                                    ui.spinner();
                                }),
                                ProgressState::Processing(text, progress) => ui.horizontal(|ui| {
                                    ui.label(text);
                                    ui.add(egui::ProgressBar::new(*progress));
                                }),
                                ProgressState::Finished => ui.horizontal(|ui| {
                                    ui.colored_label(egui::Color32::GREEN, "●");
                                    ui.label("All processing finished");
                                }),
                            });
                        },
                    );
                },
            );
        });
    }
}
