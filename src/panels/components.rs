use eframe::egui;

pub fn wide_button(name: &str, width: f32) -> impl egui::Widget {
    egui::Button::new(name).min_size(egui::Vec2::new(width, 30.0))
}
