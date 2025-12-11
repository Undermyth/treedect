#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{egui, epaint};

/// 画布状态，用于管理平移和缩放
struct CanvasState {
    /// 画布在世界空间中的偏移量（平移）
    offset: egui::Vec2,
    /// 缩放比例
    scale: f32,
    /// 鼠标拖拽的起始点
    drag_start: Option<egui::Pos2>,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            offset: egui::Vec2::ZERO,
            scale: 1.0,
            drag_start: None,
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Egui Layers Example",
        options,
        Box::new(|cc| Ok(Box::new(LayerApp::new(cc)))),
    )
}

/// 单个图层结构
struct Layer {
    name: String,
    visible: bool,
    opacity: f32, // 0.0 - 1.0
    // 原始 CPU 数据，用于后续可能的像素操作
    image_data: egui::ColorImage, 
    // GPU 纹理句柄
    texture: Option<egui::TextureHandle>,
}

impl Layer {
    fn new(name: &str, color: egui::Color32, width: usize, height: usize) -> Self {
        // 创建一个纯色的测试图片，你可以替换为加载真实的图片65
        let mut image_data = egui::ColorImage::filled([width, height], color);
        
        // 为了演示叠加，我们在中间画个洞或者画点花纹
        for y in (height/4)..(height/4*3) {
            for x in (width/4)..(width/4*3) {
                 // 设置部分像素透明，演示穿透效果
                image_data[(x, y)] = egui::Color32::TRANSPARENT; 
            }
        }

        Self {
            name: name.to_owned(),
            visible: true,
            opacity: 1.0,
            image_data,
            texture: None,
        }
    }

    /// 确保纹理已上传到 GPU
    fn texture_id(&mut self, ctx: &egui::Context) -> egui::TextureId {
        self.texture
            .get_or_insert_with(|| {
                ctx.load_texture(&self.name, self.image_data.clone(), Default::default())
            })
            .id()
    }
}

struct LayerApp {
    layers: Vec<Layer>,
    canvas_state: CanvasState,
}

impl LayerApp {
    fn new(_cc: &eframe::CreationContext) -> Self {
        // 初始化两个图层：底层红色，顶层蓝色（带透明洞）
        Self {
            layers: vec![
                Layer::new("Background (Red)", egui::Color32::from_rgb(200, 50, 50), 300, 300),
                Layer::new("Foreground (Blue)", egui::Color32::from_rgb(50, 50, 200), 300, 300),
            ],
            canvas_state: CanvasState::default(),
        }
    }
}

impl eframe::App for LayerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("layer_control").show(ctx, |ui| {
            ui.heading("Layers");
            ui.separator();

            // 倒序遍历以模拟 Photoshop 的图层列表（上面的是顶层）
            for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut layer.visible, "");
                        ui.label(format!("Layer {}: {}", i, layer.name));
                    });
                    ui.add(egui::Slider::new(&mut layer.opacity, 0.0..=1.0).text("Opacity"));
                });
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Canvas Area");
            
            // 1. 划定画布区域
            // 自适应填满整个中央面板区域
            let canvas_size = ui.available_size();
            let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());

            // 绘制背景框，方便看清画布边界
            painter.rect_stroke(response.rect, 0.0, (1.0, egui::Color32::WHITE), epaint::StrokeKind::Inside);

            // 处理拖拽 (Pan)
            if response.drag_started() {
                self.canvas_state.drag_start = response.hover_pos();
            }
            
            if response.dragged() {
                if let (Some(drag_start), Some(hover_pos)) = (self.canvas_state.drag_start, response.hover_pos()) {
                    let delta = hover_pos - drag_start;
                    self.canvas_state.offset += delta;
                    self.canvas_state.drag_start = Some(hover_pos);
                }
            }

            // 处理滚轮缩放 (Zoom)
            if let Some(hover_pos) = response.hover_pos() {
                // 获取鼠标滚轮输入
                let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 0.9 };
                    let new_scale = self.canvas_state.scale * zoom_factor;
                    
                    // 限制缩放范围，防止过大或过小
                    let new_scale = new_scale.clamp(0.1, 10.0);

                    // 关键数学：以鼠标指针为中心进行缩放
                    let screen_rect = response.rect;
                    let render_offset = hover_pos - screen_rect.min; // 鼠标相对于视口左上角的位置
                    self.canvas_state.offset -= render_offset * (zoom_factor - 1.0);
                    
                    self.canvas_state.scale = new_scale;
                }
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
                let tint = egui::Color32::from_white_alpha((layer.opacity * 255.0) as u8);

                // 3. 执行绘制，应用缩放和平移变换
                let image_size = layer.image_data.size;
                let scaled_size = egui::vec2(image_size[0] as f32, image_size[1] as f32) * self.canvas_state.scale;
                
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
                    tint
                );
            }
        });
    }
}