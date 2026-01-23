use eframe::{egui, epaint};

use crate::panels::actions;
use crate::panels::canvas;
use crate::panels::global;

pub struct Canvas {}

impl Canvas {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        // 1. 划定画布区域
        // 自适应填满整个中央面板区域
        let canvas_size = ui.available_size();
        let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());

        // 绘制背景框，方便看清画布边界
        painter.rect_stroke(
            response.rect,
            0.0,
            (1.0, egui::Color32::WHITE),
            epaint::StrokeKind::Inside,
        );

        if global.layers.len() > 0 {
            let image_size = global.layers[0].get_image_size();
            canvas::update_drag_and_zoom(
                ui,
                &response,
                &mut global.canvas_state,
                canvas_size,
                image_size,
            );
        }

        // 2. 循环绘制图层
        for layer in &mut global.layers {
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
            let scaled_size =
                egui::vec2(image_size[0] as f32, image_size[1] as f32) * global.canvas_state.scale;

            // 图片左上角在屏幕上的位置 = 视口左上角 + 当前偏移
            let image_min = response.rect.min + global.canvas_state.offset;
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

        // 4. 注册右键菜单
        response.context_menu(|ui| {
            ui.label("Canvas Menu");
            if ui.button("Delete Selection").clicked() {
                global.layers[2].remove_segment_at(global.select_pos);
            }
            if ui.button("Segment Here").clicked() {
                global.sampling_points = Some(vec![global.select_pos]);
                actions::segment_action(global, None);
                global.layers[2].rerender();
            }
            if ui.button("Reset View").clicked() {
                global.canvas_state.offset = egui::Vec2::ZERO;
                global.canvas_state.scale = 1.0;
                ui.close();
            }
            if ui.button("Zoom In").clicked() {
                global.canvas_state.scale *= 1.2;
                ui.close();
            }
            if ui.button("Zoom Out").clicked() {
                global.canvas_state.scale *= 0.8;
                ui.close();
            }
        });

        // 记录右键点击坐标
        if response.secondary_clicked() {
            let pos = response.interact_pointer_pos().unwrap_or_default();
            let canvas_pos = pos - response.rect.min;
            let canvas_pos = (canvas_pos - global.canvas_state.offset) / global.canvas_state.scale;
            global.select_pos = [canvas_pos.x as usize, canvas_pos.y as usize];
            log::info!("Selected at {:?}", global.select_pos);
        }
    }
}

pub struct Progress {}

impl Progress {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        ui.vertical_centered_justified(|ui| match &global.progress_state {
            global::ProgressState::Loading(text) => ui.horizontal(|ui| {
                ui.label(text);
                ui.spinner();
            }),
            global::ProgressState::Processing(text, progress) => ui.horizontal(|ui| {
                ui.label(text);
                ui.add(egui::ProgressBar::new(*progress).animate(true).show_percentage());
            }),
            global::ProgressState::Finished => ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::GREEN, "●");
                ui.label("All processing finished");
            }),
            global::ProgressState::Error(text) => ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::RED, "●");
                ui.label(text);
            }),
        });
    }
}

pub struct CentralPanel {
    canvas: Canvas,
    progress: Progress,
}

impl CentralPanel {
    pub fn new() -> Self {
        Self {
            canvas: Canvas::new(),
            progress: Progress::new(),
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, global: &mut global::GlobalState) {
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
                        self.canvas.ui(ctx, ui, global);
                    },
                );

                ui.separator();

                // Lower part, Progress - 10% height
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(ui.available_width(), panel_height * 0.05),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        self.progress.ui(ui, global);
                    },
                );
            },
        );
    }
}
