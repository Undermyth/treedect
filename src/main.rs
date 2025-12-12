#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{egui, epaint};

mod actions;
mod canvas;

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

struct TreeDectApp {
    layers: Vec<canvas::Layer>,
    canvas_state: canvas::CanvasState,
}

impl TreeDectApp {
    fn new(_cc: &eframe::CreationContext) -> Self {
        // 初始化两个图层：底层红色，顶层蓝色（带透明洞）
        Self {
            layers: vec![
                canvas::Layer::new(
                    "Background (Red)",
                    egui::Color32::from_rgb(200, 50, 50),
                    300,
                    300,
                ),
                canvas::Layer::new(
                    "Foreground (Blue)",
                    egui::Color32::from_rgb(50, 50, 200),
                    300,
                    300,
                ),
            ],
            canvas_state: canvas::CanvasState::default(),
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
                                if ui
                                    .add(
                                        egui::Button::new("Load Image")
                                            .min_size(egui::Vec2::new(ui.available_width(), 30.0)),
                                    )
                                    .clicked()
                                {
                                    let image_path = actions::load_image_action();
                                    if let Some(image_path) = image_path {
                                        // 加载图片
                                    }
                                }
                            });
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

                        // Upper part - 50% height
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

                        // Lower part - 50% height
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

            canvas::update_drag_and_zoom(ui, &response, &mut self.canvas_state);

            // 2. 循环绘制图层
            for layer in &mut self.layers {
                if !layer.visible || layer.opacity <= 0.0 {
                    continue;
                }

                let tex_id = layer.texture_id(ctx);

                // 计算透明度颜色混合
                // egui 的 tint 颜色会与纹理像素相乘。
                // 如果要控制整体透明度，使用带有 Alpha 的白色即可。
                let tint = egui::Color32::from_white_alpha((layer.opacity * 255.0) as u8);

                // 3. 执行绘制，应用缩放和平移变换
                let image_size = layer.get_image_size();
                let scaled_size = egui::vec2(image_size[0] as f32, image_size[1] as f32)
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
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), // UV 坐标
                    tint,
                );
            }
        });
    }
}
