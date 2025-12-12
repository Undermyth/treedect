use eframe::egui;

/// 画布状态，用于管理平移和缩放
pub struct CanvasState {
    /// 画布在世界空间中的偏移量（平移）
    pub offset: egui::Vec2,
    /// 缩放比例
    pub scale: f32,
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

pub fn update_drag_and_zoom(
    ui: &mut egui::Ui,
    response: &egui::Response,
    canvas_state: &mut CanvasState,
) {
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

            // 限制缩放范围，防止过大或过小
            let new_scale = new_scale.clamp(0.1, 10.0);

            // 关键数学：以鼠标指针为中心进行缩放
            let screen_rect = response.rect;
            let render_offset = hover_pos - screen_rect.min; // 鼠标相对于视口左上角的位置
            canvas_state.offset -= render_offset * (zoom_factor - 1.0);

            canvas_state.scale = new_scale;
        }
    }
}

/// 单个图层结构
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32, // 0.0 - 1.0
    // 原始 CPU 数据，用于后续可能的像素操作
    image_data: egui::ColorImage,
    // GPU 纹理句柄
    texture: Option<egui::TextureHandle>,
}

impl Layer {
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
            image_data,
            texture: None,
        }
    }

    pub fn from_path(name: &str, path: String) -> Result<Self, Box<dyn std::error::Error>> {
        let path = std::path::Path::new(&path);
        let image = image::ImageReader::open(path)?.decode()?;
        let size = [image.width() as _, image.height() as _];
        let image_buffer = image.to_rgba8();
        let pixels = image_buffer.as_flat_samples();
        Ok(Self {
            name: name.to_owned(),
            visible: true,
            opacity: 1.0,
            image_data: egui::ColorImage::from_rgba_premultiplied(size, pixels.as_slice()),
            texture: None,
        })
    }

    /// 确保纹理已上传到 GPU
    pub fn texture_id(&mut self, ctx: &egui::Context) -> egui::TextureId {
        self.texture
            .get_or_insert_with(|| {
                ctx.load_texture(&self.name, self.image_data.clone(), Default::default())
            })
            .id()
    }

    pub fn get_image_size(&self) -> [usize; 2] {
        self.image_data.size
    }
}
