#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{egui, epaint};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
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
        // 创建一个纯色的测试图片，你可以替换为加载真实的图片
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
}

impl LayerApp {
    fn new(_cc: &eframe::CreationContext) -> Self {
        // 初始化两个图层：底层红色，顶层蓝色（带透明洞）
        Self {
            layers: vec![
                Layer::new("Background (Red)", egui::Color32::from_rgb(200, 50, 50), 300, 300),
                Layer::new("Foreground (Blue)", egui::Color32::from_rgb(50, 50, 200), 300, 300),
            ],
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
            // 这里我们固定画布大小为 300x300，也可以做成自适应
            let canvas_size = egui::Vec2::new(300.0, 300.0);
            let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());

            // 绘制背景框，方便看清画布边界
            painter.rect_stroke(response.rect, 0.0, (1.0, egui::Color32::WHITE), epaint::StrokeKind::Inside);

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

                // 3. 执行绘制
                // 注意：这里所有图层都绘制在同一个 response.rect 上，即完全重叠
                painter.image(
                    tex_id,
                    response.rect, 
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), // UV 坐标
                    tint
                );
            }
        });
    }
}