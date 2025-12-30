use image::{GenericImage, imageops};
use image::{ImageBuffer, RgbaImage};

pub struct SAM2Batcher<'a> {
    idx: usize,
    batch_size: usize,
    patch_size: usize,
    model_rel: usize, // resolution of SAM2 model, default to 1024
    sampling_points: Vec<[usize; 2]>,
    raw_image: &'a RgbaImage,
}

impl<'a> SAM2Batcher<'a> {
    pub fn new(
        batch_size: usize,
        patch_size: usize,
        sampling_points: Vec<[usize; 2]>,
        raw_image: &'a RgbaImage,
    ) -> Self {
        Self {
            idx: 0,
            batch_size,
            patch_size,
            model_rel: 1024,
            sampling_points,
            raw_image,
        }
    }
}

impl<'a> Iterator for SAM2Batcher<'a> {
    type Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.sampling_points.len() {
            return None;
        }
        let start_idx = self.idx;
        let end_idx = (self.idx + self.batch_size).max(self.sampling_points.len());
        for i in start_idx..end_idx {
            let [x, y] = self.sampling_points[i];
            let start_x = x.saturating_sub(self.patch_size / 2);
            let start_y = y.saturating_sub(self.patch_size / 2);
            let end_x = (start_x + self.patch_size).max(self.raw_image.width() as usize);
            let end_y = (start_y + self.patch_size).max(self.raw_image.height() as usize);
            let patch = self
                .raw_image
                .sub_image(
                    start_x as u32,
                    start_y as u32,
                    (end_x - start_x) as u32,
                    (end_y - start_y) as u32,
                )
                .to_image();
            let patch = imageops::resize(
                &patch,
                self.model_rel as u32,
                self.model_rel as u32,
                image::imageops::FilterType::Lanczos3, // TODO: or CatmullRom?
            );
            self.idx += 1;
            return Some(patch);
        }
        None
    }
}
