use eframe::egui;
use std::sync::{Arc, Mutex};

use crate::panels::palette;

/// 画布状态，用于管理平移和缩放
pub struct CanvasState {
    /// 画布在世界空间中的偏移量（平移）
    pub offset: egui::Vec2,
    /// 缩放比例
    pub scale: f32,
    /// 鼠标拖拽的起始点
    drag_start: Option<egui::Pos2>,
    /// 是否初始化
    initialized: bool,
    /// 加载图像之后自动缩放的偏移
    pub initial_offset: egui::Vec2,
    /// 加载图像之后自动缩放的比例
    pub initial_scale: f32,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            offset: egui::Vec2::ZERO,
            scale: 1.0,
            drag_start: None,
            initialized: false,
            initial_offset: egui::Vec2::ZERO,
            initial_scale: 1.0,
        }
    }
}

pub fn update_drag_and_zoom(
    ui: &mut egui::Ui,
    response: &egui::Response,
    canvas_state: &mut CanvasState,
    canvas_size: egui::Vec2,
    image_size: [usize; 2],
) {
    if !canvas_state.initialized {
        let initial_scale = (canvas_size.x as f32 / image_size[0] as f32)
            .min(canvas_size.y as f32 / image_size[1] as f32);
        let initial_offset = egui::Vec2::new((canvas_size.x - canvas_size.y) / 2.0, 0.0);
        canvas_state.scale = initial_scale;
        canvas_state.offset = initial_offset;
        canvas_state.initialized = true;
        canvas_state.initial_scale = initial_scale;
        canvas_state.initial_offset = initial_offset;
        log::info!("canvas size: {canvas_size:?}");
        log::info!("initial_scale: {initial_scale:?}");
        log::info!("initial_offset: {initial_offset:?}");
        log::info!("image_size: {image_size:?}");
    }
    // 处理拖拽 (Pan)
    if response.drag_started() {
        canvas_state.drag_start = response.hover_pos();
    }

    if response.dragged() {
        if let (Some(drag_start), Some(hover_pos)) = (canvas_state.drag_start, response.hover_pos())
        {
            let delta = hover_pos - drag_start;
            canvas_state.offset += delta;
            canvas_state.drag_start = Some(hover_pos);
        }
    }

    // 处理滚轮缩放 (Zoom)
    if let Some(hover_pos) = response.hover_pos() {
        // 获取鼠标滚轮输入
        let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
        if scroll_delta != 0.0 {
            let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 0.9 };
            let new_scale = canvas_state.scale * zoom_factor;
            // let new_scale = canvas_state.scale + (zoom_factor - 1.0);

            // 限制缩放范围，防止过大或过小
            let new_scale = new_scale.clamp(0.1, 10.0);

            // 关键数学：以鼠标指针为中心进行缩放
            let screen_rect = response.rect;
            let render_offset = hover_pos - screen_rect.min; // 鼠标相对于视口左上角的位置
            canvas_state.offset =
                zoom_factor * canvas_state.offset + (1.0 - zoom_factor) * render_offset;

            canvas_state.scale = new_scale;
        }
    }
}

#[derive(Debug, Clone)]
pub enum LayerImage {
    RGBAImage(image::RgbaImage),
    RGBImage(image::RgbImage),
    EguiImage(egui::ColorImage),
}

impl LayerImage {
    pub fn from_palette(palette: &palette::Palette) -> Self {
        let mut image_data =
            egui::ColorImage::filled([palette.size, palette.size], egui::Color32::TRANSPARENT);
        for ((y, x), index) in palette.map.indexed_iter() {
            if *index != 0 {
                image_data[(x, y)] = egui::Color32::from_rgb(
                    palette.color_map[*index].r,
                    palette.color_map[*index].g,
                    palette.color_map[*index].b,
                );
            }
        }
        LayerImage::EguiImage(image_data)
    }
    pub fn from_palette_cluster(palette: Arc<Mutex<palette::Palette>>) -> Self {
        let palette = palette.lock().unwrap();
        let mut image_data =
            egui::ColorImage::filled([palette.size, palette.size], egui::Color32::TRANSPARENT);
        for ((y, x), index) in palette.map.indexed_iter() {
            if *index != 0 {
                image_data[(x, y)] = egui::Color32::from_rgb(
                    palette.color_map[palette.cluster_map[index - 1]].r,
                    palette.color_map[palette.cluster_map[index - 1]].g,
                    palette.color_map[palette.cluster_map[index - 1]].b,
                );
            }
        }
        LayerImage::EguiImage(image_data)
    }
    pub fn get_pixel(&self, x: usize, y: usize) -> palette::RGBPixel {
        match self {
            LayerImage::RGBAImage(image) => {
                let pixel = &image[(x as u32, y as u32)];
                palette::RGBPixel {
                    r: pixel[0],
                    g: pixel[1],
                    b: pixel[2],
                }
            }
            LayerImage::RGBImage(image) => {
                let pixel = &image[(x as u32, y as u32)];
                palette::RGBPixel {
                    r: pixel[0],
                    g: pixel[1],
                    b: pixel[2],
                }
            }
            LayerImage::EguiImage(image) => {
                let pixel = &image[(x, y)];
                palette::RGBPixel {
                    r: pixel.r(),
                    g: pixel.g(),
                    b: pixel.b(),
                }
            }
        }
    }
}

/// 单个图层结构
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32, // 0.0 - 1.0
    /// raw image is **read only**. Once the image is uploaded to GPU,
    /// the texture is directly handled by `texture` handler.
    /// for image layer, the ownership will be transfered to global state,
    /// and `raw_image` will be set to `None` soon after creation.
    pub raw_image: Option<LayerImage>,
    /// GPU 纹理句柄
    texture: Option<egui::TextureHandle>,
    // below for editable layers. If the layer is not editable, palette
    // and other data structure will not be maintained.
    pub editable: bool,
}

impl Layer {
    #[allow(dead_code)]
    pub fn new(name: &str, color: egui::Color32, width: usize, height: usize) -> Self {
        // 创建一个纯色的测试图片，你可以替换为加载真实的图片65
        let mut image_data = egui::ColorImage::filled([width, height], color);

        // 为了演示叠加，我们在中间画个洞或者画点花纹
        for y in (height / 4)..(height / 4 * 3) {
            for x in (width / 4)..(width / 4 * 3) {
                // 设置部分像素透明，演示穿透效果
                image_data[(x, y)] = egui::Color32::TRANSPARENT;
            }
        }

        Self {
            name: name.to_owned(),
            visible: true,
            opacity: 1.0,
            raw_image: Some(LayerImage::EguiImage(image_data)),
            texture: None,
            editable: false,
        }
    }

    // 异步版本的图像加载
    // create image layer
    pub fn from_path(
        name: String,
        path: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = std::path::Path::new(&path);
        let image = image::ImageReader::open(path)?.decode()?;
        let image_buffer = image.to_rgb8();
        Ok::<Self, Box<dyn std::error::Error + Send + Sync>>(Self {
            name: name.to_owned(),
            visible: true,
            opacity: 1.0,
            raw_image: Some(LayerImage::RGBImage(image_buffer)),
            // image_data: egui::ColorImage::from_rgba_premultiplied(size, pixels.as_slice()),
            texture: None,
            editable: false,
        })
    }

    pub fn from_palette(name: String, palette: &palette::Palette) -> Self {
        let image = LayerImage::from_palette(palette);
        let image = if palette.debug {
            if let LayerImage::EguiImage(mut image_data) = image {
                for bbox in palette.bboxes.iter() {
                    let [y, x, size] = bbox;
                    Layer::cpu_draw_circle(&mut image_data, &[*y, *x], 5, egui::Color32::GREEN);
                    Layer::cpu_draw_circle(
                        &mut image_data,
                        &[y + size, x + size],
                        5,
                        egui::Color32::LIGHT_BLUE,
                    );
                }
                LayerImage::EguiImage(image_data)
            } else {
                image
            }
        } else {
            image
        };
        Self {
            name,
            visible: true,
            opacity: 0.7,
            raw_image: Some(image),
            texture: None,
            editable: true,
        }
    }

    pub fn from_palette_cluster(name: String, palette: Arc<Mutex<palette::Palette>>) -> Self {
        let image = LayerImage::from_palette_cluster(palette);
        Self {
            name,
            visible: true,
            opacity: 0.7,
            raw_image: Some(image),
            texture: None,
            editable: true,
        }
    }

    fn cpu_draw_circle(
        image: &mut egui::ColorImage,
        center: &[usize; 2],
        radius: usize,
        color: egui::Color32,
    ) {
        let center_x = center[0];
        let center_y = center[1];
        let width = image.width();
        let height = image.height();
        for x in (center_x.saturating_sub(radius))..(center_x + radius).min(width) {
            for y in (center_y.saturating_sub(radius))..(center_y + radius).min(height) {
                if ((x as i32 - center_x as i32).pow(2) + (y as i32 - center_y as i32).pow(2))
                    as usize
                    <= radius.pow(2)
                {
                    image[(x, y)] = color;
                }
            }
        }
    }

    fn cpu_draw_center_square(image: &mut egui::ColorImage, length: usize, width: usize) {
        let img_width = image.width();
        let img_height = image.height();

        // Calculate the center of the image
        let center_x = img_width / 2;
        let center_y = img_height / 2;

        // Calculate the top-left corner of the square
        let start_x = center_x.saturating_sub(length / 2);
        let start_y = center_y.saturating_sub(length / 2);

        // Calculate the bottom-right corner of the square
        let end_x = (start_x + length).min(img_width);
        let end_y = (start_y + length).min(img_height);

        // Draw the square frame with the specified stroke width
        for y in start_y..end_y {
            for x in start_x..end_x {
                // Check if this pixel is on the border of the square
                if x < start_x + width
                    || x >= end_x - width
                    || y < start_y + width
                    || y >= end_y - width
                {
                    // Only set the pixel if it's within image bounds
                    if x < img_width && y < img_height {
                        image[(x, y)] = egui::Color32::RED;
                    }
                }
            }
        }
    }

    pub fn from_sampling_points(
        sampling_points: &Vec<[usize; 2]>,
        width: usize,
        height: usize,
        rel: usize,
    ) -> Self {
        let mut image_data = egui::ColorImage::filled([width, height], egui::Color32::TRANSPARENT);
        for point in sampling_points {
            Layer::cpu_draw_circle(&mut image_data, point, 15, egui::Color32::RED);
        }
        Layer::cpu_draw_center_square(&mut image_data, rel, 10);
        Self {
            name: "sampling points".to_string(),
            visible: true,
            opacity: 1.0,
            raw_image: Some(LayerImage::EguiImage(image_data)),
            texture: None,
            editable: false,
        }
    }

    pub fn rerender(&mut self, image: LayerImage) {
        if let LayerImage::EguiImage(image_data) = image {
            self.texture
                .as_mut()
                .unwrap()
                .set(image_data, egui::TextureOptions::default());
        }
    }

    /// 确保纹理已上传到 GPU
    pub fn texture_id(&mut self, ctx: &egui::Context) -> egui::TextureId {
        if let Some(texture) = &self.texture {
            return texture.id();
        }
        let egui_image_data = if let Some(LayerImage::RGBImage(image_buffer)) = &self.raw_image {
            egui::ColorImage::from_rgb(
                [
                    image_buffer.width() as usize,
                    image_buffer.height() as usize,
                ],
                image_buffer.as_flat_samples().as_slice(),
            )
        } else if let Some(LayerImage::RGBAImage(image_buffer)) = &self.raw_image {
            egui::ColorImage::from_rgba_premultiplied(
                [
                    image_buffer.width() as usize,
                    image_buffer.height() as usize,
                ],
                image_buffer.as_flat_samples().as_slice(),
            )
        } else if let Some(LayerImage::EguiImage(image_buffer)) = &self.raw_image {
            image_buffer.clone()
        } else {
            egui::ColorImage::example() // when raw_image is None, return a default ColorImage
        };
        self.texture
            .get_or_insert_with(|| {
                ctx.load_texture(&self.name, egui_image_data, Default::default())
            })
            .id()
    }

    pub fn get_image_size(&self) -> [usize; 2] {
        if let Some(LayerImage::RGBImage(image_buffer)) = &self.raw_image {
            [
                image_buffer.width() as usize,
                image_buffer.height() as usize,
            ]
        } else if let Some(LayerImage::RGBAImage(image_buffer)) = &self.raw_image {
            [
                image_buffer.width() as usize,
                image_buffer.height() as usize,
            ]
        } else if let Some(LayerImage::EguiImage(image_buffer)) = &self.raw_image {
            image_buffer.size
        } else {
            [0, 0]
        }
    }
}
