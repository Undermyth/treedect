use eframe::egui;
use std::sync::mpsc::Sender;

pub fn load_image_action(ctx: egui::Context, sender: Sender<String>) {
    let task = rfd::AsyncFileDialog::new()
        .add_filter("Image Files", &["jpg", "png", "jpeg", "bmp", "tif", "tiff"])
        .pick_file();
    std::thread::spawn(move || {
        let future = async move {
            if let Some(file_handle) = task.await {
                if let Some(path) = file_handle.path().to_str() {
                    let _ = sender.send(path.to_string());
                    ctx.request_repaint();
                }
            }
        };
        futures::executor::block_on(future);
    });
}
