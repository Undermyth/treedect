use eframe::egui;

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
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            offset: egui::Vec2::ZERO,
            scale: 1.0,
            drag_start: None,
            initialized: false,
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
        log::info!("initial_scale: {initial_scale:?}");
        log::info!("initial_offset: {initial_offset:?}");
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

/// 单个图层结构
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32, // 0.0 - 1.0
    pub raw_image: Option<image::RgbaImage>,
    // 原始 CPU 数据，用于后续可能的像素操作
    pub image_data: egui::ColorImage,
    // GPU 纹理句柄
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
            raw_image: None,
            image_data,
            texture: None,
            editable: false,
        }
    }

    // 异步版本的图像加载
    // create image layer
    pub async fn from_path(
        name: String,
        path: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 在后台线程中执行阻塞的文件I/O操作
        let layer = tokio::task::spawn_blocking(move || {
            let path = std::path::Path::new(&path);
            let image = image::ImageReader::open(path)?.decode()?;
            let size = [image.width() as _, image.height() as _];
            let image_buffer = image.to_rgba8();
            let pixels = image_buffer.as_flat_samples();
            let raw_image = image_buffer.clone();
            Ok::<Self, Box<dyn std::error::Error + Send + Sync>>(Self {
                name: name.to_owned(),
                visible: true,
                opacity: 1.0,
                raw_image: Some(raw_image),
                image_data: egui::ColorImage::from_rgba_premultiplied(size, pixels.as_slice()),
                texture: None,
                editable: false,
            })
        })
        .await??;

        Ok(layer)
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
            raw_image: None,
            image_data,
            texture: None,
            editable: false,
        }
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
