use image::RgbaImage;
use image::{GenericImageView, imageops};
use ndarray::{Array1, Array4};

pub struct SAM2Batcher<'a> {
    idx: usize,
    batch_size: usize,
    patch_size: usize,
    model_rel: usize, // resolution of SAM2 model, default to 1024
    sampling_points: Vec<[usize; 2]>,
    raw_image: &'a RgbaImage,
    mean: Array1<f32>,
    std: Array1<f32>,
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
            mean: Array1::from_vec(vec![0.485, 0.456, 0.406]),
            std: Array1::from_vec(vec![0.229, 0.224, 0.225]),
        }
    }
}

impl<'a> Iterator for SAM2Batcher<'a> {
    type Item = Array4<f32>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.sampling_points.len() {
            return None;
        }
        let start_idx = self.idx;
        let end_idx = (self.idx + self.batch_size).min(self.sampling_points.len());
        let actual_batch_size = end_idx - start_idx;
        let start_time = std::time::Instant::now();
        let mut batch = Array4::<f32>::default((actual_batch_size, self.model_rel, self.model_rel, 3));
        let allocate_time = start_time.elapsed();
        log::info!("Allocate time: {:?}", allocate_time);
        for i in start_idx..end_idx {
            let [x, y] = self.sampling_points[i];
            let start_x = x.saturating_sub(self.patch_size / 2);
            let start_y = y.saturating_sub(self.patch_size / 2);
            let end_x = (start_x + self.patch_size).min(self.raw_image.width() as usize);
            let end_y = (start_y + self.patch_size).min(self.raw_image.height() as usize);
            let patch = self
                .raw_image
                .view(
                    start_x as u32,
                    start_y as u32,
                    (end_x - start_x) as u32,
                    (end_y - start_y) as u32,
                )
                .to_image();
            let start_time = std::time::Instant::now();
            let patch = imageops::resize(
                &patch,
                self.model_rel as u32,
                self.model_rel as u32,
                image::imageops::FilterType::Nearest, // TODO: or CatmullRom?
            );
            let resize_time = start_time.elapsed();
            log::info!("Resize time: {:?}", resize_time);
            let start_time = std::time::Instant::now();
            for x in 0..self.patch_size {
                for y in 0..self.patch_size {
                    for c in 0..3 {
                        batch[[i - start_idx, x, y, c]] = patch.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                    }
                }
            }
            let process_time = start_time.elapsed();
            log::info!("Process time: {:?}", process_time);
        }
        batch = (batch - self.mean.to_shape([1, 1, 1, 3]).unwrap())
            / self.std.to_shape([1, 1, 1, 3]).unwrap();
        batch = batch.permuted_axes([0, 3, 1, 2]).to_owned();
        self.idx += actual_batch_size;
        log::info!("{}", self.idx);
        return Some(batch);
    }
}
