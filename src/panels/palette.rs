use ndarray::Array2;
use rand::Rng;

static MAX_PALETTE_SIZE: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct RGBPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl RGBPixel {
    pub fn default() -> Self {
        Self { r: 0, g: 0, b: 0 }
    }
}

#[derive(Debug)]
pub struct Palette {
    pub size: usize,
    pub num_patches: usize,
    pub map: Array2<usize>,
    pub color_map: [RGBPixel; MAX_PALETTE_SIZE],
    pub bboxes: Vec<[usize; 3]>,
    pub valid: Vec<bool>,
}

impl Palette {
    pub fn new(size: usize) -> Self {
        let mut rng = rand::rng();
        let mut color_map = [RGBPixel::default(); MAX_PALETTE_SIZE];
        for i in 0..MAX_PALETTE_SIZE {
            color_map[i].r = rng.random_range(0..255);
            color_map[i].g = rng.random_range(0..127);
            color_map[i].b = rng.random_range(0..255);
        }
        Self {
            size,
            num_patches: 0,
            map: Array2::zeros([size, size]),
            color_map,
            bboxes: Vec::new(),
            valid: Vec::new(),
        }
    }
}
