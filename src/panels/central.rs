use eframe::{egui, epaint};

use crate::panels::actions;
use crate::panels::canvas;
use crate::panels::global;

use egui_plot::Plot;
use egui_plot::PlotPoint;

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

        // 使用 Area 容器在 painter 区域上重叠显示 plot，并处理交互
        let plot_area_response = egui::Area::new(egui::Id::new("overlay_plot"))
            .fixed_pos(response.rect.min)
            .interactable(true) // 设为可交互，以接收鼠标事件
            .show(ctx, |ui| {
                ui.set_width(response.rect.width());
                ui.set_height(response.rect.height());

                let my_plot = Plot::new("My Plot")
                    // .legend(Legend::default())
                    .show_background(false)
                    .show_axes([false, false])
                    .show_grid([false, false])
                    .allow_zoom([false, false])
                    .allow_scroll([false, false])
                    .allow_drag([false, false])
                    .invert_y(true)
                    .default_x_bounds(0.0, response.rect.width() as f64)
                    .default_y_bounds(0.0, response.rect.height() as f64);

                let plot_response = my_plot.show(ui, |plot_ui| {
                    if global.palette.is_none() || !global.params.show_cluster_ids {
                        return;
                    }
                    let palette = global.palette.as_ref().unwrap();
                    let palette = palette.clone();
                    let palette = palette.lock().unwrap();
                    if palette.num_clusters == 0 {
                        return;
                    }
                    for (i, cluster_id) in palette.cluster_map.iter().enumerate() {
                        if *cluster_id == 0 {
                            continue;
                        }
                        let [x, y, size] = palette.bboxes[i];
                        let mut pos_on_canvas =
                            egui::Vec2::new((x + size / 2) as f32, (y + size / 2) as f32);
                        pos_on_canvas =
                            pos_on_canvas * global.canvas_state.scale + global.canvas_state.offset;
                        let font_size =
                            18.0 * global.canvas_state.scale / global.canvas_state.initial_scale;
                        let text = egui::RichText::new(cluster_id.to_string())
                            .size(font_size)
                            .color(egui::Color32::WHITE);
                        plot_ui.text(egui_plot::Text::new(
                            cluster_id.to_string(),
                            PlotPoint::new(pos_on_canvas.x, pos_on_canvas.y),
                            text,
                        ))
                    }
                });

                // 返回 plot 的 response 用于交互处理
                plot_response.response
            })
            .inner;

        // 优先使用 plot_area_response 处理交互，如果不可用则回退到 painter 的 response
        let interaction_response = if plot_area_response.hovered()
            || plot_area_response.clicked()
            || plot_area_response.dragged()
        {
            &plot_area_response
        } else {
            &response
        };

        // 处理拖拽和缩放交互
        if global.layers.len() > 0 {
            let image_size = global.get_layer(global::LayerType::Image).get_image_size();
            canvas::update_drag_and_zoom(
                ui,
                interaction_response,
                &mut global.canvas_state,
                canvas_size,
                image_size,
            );
        }

        // 4. 注册右键菜单 - 使用 plot_area_response
        plot_area_response.context_menu(|ui| {
            ui.label("Canvas Menu");
            if ui.button("Delete Selection").clicked() {
                let palette = global.palette.as_ref().unwrap().clone();
                let mut palette = palette.lock().unwrap();
                palette.remove_segment_at(global.select_pos);
                global
                    .get_layer(global::LayerType::Segmentation)
                    .rerender(canvas::LayerImage::from_palette(&palette));
            }
            if ui.button("Segment Here").clicked() {
                global.sampling_points = Some(vec![global.select_pos]);
                actions::point_segment_action(global);
                let palette = global.palette.as_ref().unwrap().clone();
                let palette = palette.lock().unwrap();
                global
                    .get_layer(global::LayerType::Segmentation)
                    .rerender(canvas::LayerImage::from_palette(&palette));
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
        if plot_area_response.secondary_clicked() {
            let pos = plot_area_response
                .interact_pointer_pos()
                .unwrap_or_default();
            let canvas_pos = pos - response.rect.min;
            let canvas_pos = (canvas_pos - global.canvas_state.offset) / global.canvas_state.scale;
            global.select_pos = [canvas_pos.x as usize, canvas_pos.y as usize];
            if global.layers.len() >= 3 {
                log::info!(
                    "Selected at {:?}, palette index: {:?}",
                    global.select_pos,
                    global.palette.as_ref().unwrap().lock().unwrap().map
                        [(global.select_pos[1], global.select_pos[0])]
                );
            } else {
                log::info!("Selected at {:?}", global.select_pos);
            }
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
                ui.add(
                    egui::ProgressBar::new(*progress)
                        .animate(true)
                        .show_percentage(),
                );
            }),
            global::ProgressState::Finished(text) => ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::GREEN, "●");
                ui.label(text);
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
