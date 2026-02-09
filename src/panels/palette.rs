use ndarray::Array2;
use rand::Rng;
use std::collections::HashSet;

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
    pub areas: Vec<usize>,
    pub n_grids: usize,
    pub grids: Vec<usize>, // used to calculate the importance score
    pub debug: bool,
    pub cluster_map: Vec<usize>, // map from patch index (start from 0) to cluster index (start from 1)
}

impl Palette {
    pub fn new(size: usize, n_grids: usize) -> Self {
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
            areas: Vec::new(),
            n_grids: n_grids,
            grids: Vec::new(),
            debug: false,
            cluster_map: Vec::new(),
        }
    }
    pub fn set_cluster_map(&mut self, cluster: Vec<Vec<usize>>) {
        // see `cluster.py` for the details
        self.cluster_map.resize(self.num_patches, 0);
        for (i, patch_ids) in cluster.iter().enumerate() {
            for patch_id in patch_ids {
                self.cluster_map[*patch_id - 1] = i + 1;
            }
        }
    }
    pub fn get_statistics(&mut self) {
        self.areas.resize(self.num_patches, 0);
        self.grids.resize(self.num_patches, 0);
        let mut grid_list = Vec::<HashSet<usize>>::new();
        let grid_size = self.size / self.n_grids;
        grid_list.resize(self.num_patches, HashSet::<usize>::new());
        for ((y, x), index) in self.map.indexed_iter() {
            if *index != 0 {
                let grid_id = (y / grid_size) * self.n_grids + (x / grid_size);
                grid_list[*index - 1].insert(grid_id);
                self.areas[*index - 1] += 1;
            }
        }
        for (i, set) in grid_list.iter().enumerate() {
            self.grids[i] = set.len();
        }
    }
}
