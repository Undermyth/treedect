use fast_image_resize as fr;
use fast_image_resize::images::Image;
use image::RgbImage;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array1, Array2, Array3, Array4, array, s};

pub struct SAM2Batch {
    // [B, C, H, W] where H = W = 1024
    pub image: Array4<f32>,
    /// [B, 2], left up corner [x, y] of the patches.
    /// The coordinate is in the format [width, height], aligned with the image coordinate system.
    /// The whole patch will be [x..x+patch_size, y..y+patch_size]
    pub coordinates: Array2<usize>,
    /// [B, 2], the coordinate of sampling points in the image.
    /// ***The coordinate is in the format [height, width] to align with DL models.***
    /// The sampling points are recorded before scaling. All scaling will be processed
    /// within SAM2 model.
    pub sampling_coords: Array2<f32>,
    /// original size of patch
    pub original_size: usize,
}

pub struct SAM2Batcher<'a> {
    idx: usize,
    batch_size: usize,
    patch_size: usize,
    model_rel: usize, // resolution of SAM2 model, default to 1024
    sampling_points: Vec<[usize; 2]>,
    raw_image: &'a RgbImage,
    mean: Array1<f32>,
    std: Array1<f32>,
}

impl<'a> SAM2Batcher<'a> {
    pub fn new(
        batch_size: usize,
        patch_size: usize,
        sampling_points: Vec<[usize; 2]>,
        raw_image: &'a RgbImage,
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
    type Item = SAM2Batch;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.sampling_points.len() {
            return None;
        }
        let start_idx = self.idx;
        let end_idx = (self.idx + self.batch_size).min(self.sampling_points.len());
        let actual_batch_size = end_idx - start_idx;
        let mut batch =
            Array4::<f32>::uninit((actual_batch_size, self.model_rel, self.model_rel, 3));
        let mut sampling_coords = Array2::<f32>::uninit((actual_batch_size, 2));
        let mut coordinates = Array2::<usize>::uninit((actual_batch_size, 2));
        for i in start_idx..end_idx {
            let [x, y] = self.sampling_points[i];
            let mut start_x = x.saturating_sub(self.patch_size / 2);
            let mut start_y = y.saturating_sub(self.patch_size / 2);
            let end_x = (start_x + self.patch_size).min(self.raw_image.width() as usize);
            let end_y = (start_y + self.patch_size).min(self.raw_image.height() as usize);
            if end_x == self.raw_image.width() as usize {
                start_x = end_x - self.patch_size;
            }
            if end_y == self.raw_image.height() as usize {
                start_y = end_y - self.patch_size;
            }
            array![start_y, start_x]
                .view()
                .assign_to(coordinates.slice_mut(s![i - start_idx, ..]));
            array![(y - start_y) as f32, (x - start_x) as f32]
                .view()
                .assign_to(sampling_coords.slice_mut(s![i - start_idx, ..]));
            let patch = self
                .raw_image
                .view(
                    start_x as u32,
                    start_y as u32,
                    (end_x - start_x) as u32,
                    (end_y - start_y) as u32,
                )
                .to_image();
            let patch = DynamicImage::ImageRgb8(patch);
            let mut dst_image = Image::new(
                self.model_rel as u32,
                self.model_rel as u32,
                fr::PixelType::U8x3,
            );
            let mut resizer = fr::Resizer::new();
            let _ = resizer.resize(&patch, &mut dst_image, None);
            let shape = (self.model_rel, self.model_rel, 3);
            let mut sample = Array3::from_shape_vec(
                shape,
                dst_image.into_vec().into_iter().map(|b| b as f32).collect(),
            )
            .unwrap();
            sample = (sample / 255.0 - self.mean.to_shape([1, 1, 3]).unwrap())
                / self.std.to_shape([1, 1, 3]).unwrap();
            sample
                .view()
                .assign_to(batch.slice_mut(s![i - start_idx, .., .., ..]));
        }
        self.idx += actual_batch_size;
        // log::info!("Proceeding batch to {}", self.idx);
        unsafe {
            return Some(SAM2Batch {
                image: batch
                    .assume_init()
                    .permuted_axes([0, 3, 1, 2])
                    .as_standard_layout()
                    .to_owned(),
                coordinates: coordinates.assume_init().to_owned(),
                sampling_coords: sampling_coords.assume_init().to_owned(),
                original_size: self.patch_size,
            });
        }
    }
}
