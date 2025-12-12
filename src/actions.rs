pub fn load_image_action() -> Option<String> {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Image Files", &["jpg", "png", "jpeg", "bmp", "tif", "tiff"])
        .pick_file()
    {
        log::info!("Selected file: {:?}", path);
        Some(path.display().to_string())
    } else {
        None
    }
}
