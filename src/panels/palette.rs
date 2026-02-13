use ndarray::Array2;
use rand::Rng;
use std::collections::HashMap;
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
    pub max_patch_id: usize,
    pub num_clusters: usize,
    pub map: Array2<usize>,
    pub color_map: [RGBPixel; MAX_PALETTE_SIZE],
    pub bboxes: Vec<[usize; 3]>,
    pub valid: Vec<bool>,
    pub areas: Vec<usize>,
    pub n_grids: Option<usize>,
    pub grids: Vec<usize>, // used to calculate the importance score
    pub debug: bool,
    pub cluster_map: Vec<usize>, // map from patch index (start from 0) to cluster index (start from 1)
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
            max_patch_id: 0,
            num_clusters: 0,
            map: Array2::zeros([size, size]),
            color_map,
            bboxes: Vec::new(),
            valid: Vec::new(),
            areas: Vec::new(),
            n_grids: None,
            grids: Vec::new(),
            debug: false,
            cluster_map: Vec::new(),
        }
    }

    /// the getter will translate the index.
    /// the `segment_id` should start from 1.
    pub fn get_area(&self, segment_id: usize) -> usize {
        return self.areas[segment_id - 1];
    }

    pub fn set_cluster_map(&mut self, cluster: Vec<Vec<usize>>) {
        // see `cluster.py` for the details
        self.cluster_map.resize(self.max_patch_id, 0);
        self.num_clusters = cluster.len();
        for (i, patch_ids) in cluster.iter().enumerate() {
            for patch_id in patch_ids {
                self.cluster_map[*patch_id - 1] = i + 1;
            }
        }
    }

    /// The statistics include `bboxes` and `grids`. These features
    /// are not responsively maintained, and will only update according
    /// to the state of palette map when the function is called.
    ///
    /// Note that `n_grids` should be set before calling the method.
    /// If not set, it will give a warning and use the default of 2.
    ///
    /// Note that empty segments should be cleared before calling this.
    pub fn get_statistics(&mut self) {
        // auxiliary varibles for grids
        let n_grids = match self.n_grids {
            Some(n_grids) => n_grids,
            None => {
                log::warn!("n_grids is not set in palette. Use default of 2.");
                2
            }
        };
        self.grids.resize(self.max_patch_id, 0);
        let mut grid_list = Vec::<HashSet<usize>>::new();
        let grid_size = self.size / n_grids;
        grid_list.resize(self.max_patch_id, HashSet::<usize>::new());

        // auxiliary variables for bboxs
        self.bboxes.resize(self.max_patch_id, [0, 0, 0]);
        let mut start_x_idx = vec![self.size; self.max_patch_id];
        let mut start_y_idx = vec![self.size; self.max_patch_id];
        let mut end_x_idx = vec![0; self.max_patch_id];
        let mut end_y_idx = vec![0; self.max_patch_id];

        // scan the palette
        for ((y, x), index) in self.map.indexed_iter() {
            if *index != 0 {
                // maintain grids
                let grid_id = (y / grid_size) * n_grids + (x / grid_size);
                grid_list[*index - 1].insert(grid_id);

                // maintain bboxs
                if x < start_x_idx[*index - 1] {
                    start_x_idx[*index - 1] = x;
                }
                if x > end_x_idx[*index - 1] {
                    end_x_idx[*index - 1] = x;
                }
                if y < start_y_idx[*index - 1] {
                    start_y_idx[*index - 1] = y;
                }
                if y > end_y_idx[*index - 1] {
                    end_y_idx[*index - 1] = y;
                }
            }
        }

        // calculate grids
        for (i, set) in grid_list.iter().enumerate() {
            self.grids[i] = set.len();
        }

        // calculate bboxs
        for index in 1..=self.max_patch_id {
            let real_index = index - 1;
            if !self.valid[real_index] {
                continue;
            }
            let width = end_x_idx[real_index] - start_x_idx[real_index] + 1;
            let height = end_y_idx[real_index] - start_y_idx[real_index] + 1;
            let mut start_y_coord = start_y_idx[real_index];
            let mut start_x_coord = start_x_idx[real_index];
            if width > height {
                let bbox_size = width;
                let height_offset = (width - height) / 2;
                start_y_coord = start_y_coord.saturating_sub(height_offset);
                if start_y_coord + bbox_size > self.size {
                    start_y_coord = self.size - bbox_size;
                }
                self.bboxes[real_index] = [start_x_coord, start_y_coord, bbox_size];
            } else {
                let bbox_size = height;
                let width_offset = (height - width) / 2;
                start_x_coord = start_x_coord.saturating_sub(width_offset);
                if start_x_coord + bbox_size > self.size {
                    start_x_coord = self.size - bbox_size;
                }
                self.bboxes[real_index] = [start_x_coord, start_y_coord, bbox_size]
            }
        }
    }

    /// During segmentation and manual modification, some segments will completely
    /// vanish from palette. Since area is responsively maintained, we will update
    /// the valid flag of these segments to 0.
    ///
    /// We use lazy deletion in the palette, the arrays are **never** actually
    /// deleted. Since only 1D arrays are used with length of ~1000, lazy deletion
    /// is fast and efficient enough.
    pub fn clear_empty_segments(&mut self) {
        for (i, area) in self.areas.iter().enumerate() {
            if *area == 0 {
                self.valid[i] = false;
            }
        }
    }

    pub fn clear_small_segments(&mut self, thr: usize) {
        // Collect indices of segments to remove first to avoid borrow conflicts

        let to_remove: Vec<usize> = self
            .areas
            .iter()
            .enumerate()
            .filter(|(i, area)| self.valid[*i] && **area < thr)
            .map(|(i, _)| i + 1)
            .collect();

        // Now remove the segments

        for segment_id in to_remove {
            self.remove_segment(segment_id);
        }
    }

    pub fn get_id_at_position(&self, pos: [usize; 2]) -> Option<usize> {
        let index = self.map[(pos[1], pos[0])];
        if index != 0 {
            return Some(index);
        } else {
            return None;
        }
    }
    pub fn remove_segment_at(&mut self, pos: [usize; 2]) {
        let segment_id = self.get_id_at_position(pos);
        if let Some(segment_id) = segment_id {
            let new_map = self.map.mapv(|x| if x == segment_id { 0 } else { x });
            self.map = new_map;
            self.valid[segment_id - 1] = false;
        }
    }
    pub fn remove_segment(&mut self, segment_id: usize) {
        let new_map = self.map.mapv(|x| if x == segment_id { 0 } else { x });
        self.map = new_map;
        self.valid[segment_id - 1] = false;
    }

    /// Detect whether a given segment is overlaped with existing segments, and how.
    /// The segment is described by a `mask`. The `mask` is a subregion of the palette,
    /// the the position fixed by `coord` and `size`.
    ///
    /// # Arguments
    /// * `mask` - ndarray describing the mask.
    /// * `coord` - left upper corner of the mask on palette. Given in [height, width]
    /// * `size` - size of the mask.
    ///
    /// # Returns
    /// A HashMap. The keys of hashmap describe the segments that overlap with the given
    /// mask. The values are the overlapping area by pixels.
    pub fn detect_overlap(
        &self,
        mask: &Array2<usize>,
        coord: [usize; 2],
        size: usize,
    ) -> HashMap<usize, u32> {
        let mut result = HashMap::new();
        let [x, y] = coord;
        for i in 0..size {
            for j in 0..size {
                if mask[(i, j)] == 0 {
                    continue;
                }
                let palette_x = x + i;
                let palette_y = y + j;
                let index = self.map[(palette_x, palette_y)];
                if index != 0 {
                    let count = result.entry(index).or_insert(0);
                    *count += 1;
                }
            }
        }
        return result;
    }

    /// Add a segment to the palette. Note that only the areas are responsively updated.
    /// If the segment overlap with other segments, the area of other segments will
    /// also be updated. The grids are not maintained in a responsive way.
    /// The segment id of the new segment is automatically allocated incrementally.
    ///
    /// # Arguments:
    /// * `mask` - ndarray describing the mask.
    /// * `coord` - left upper corner of the mask on palette. Given in [height, width]
    /// * `size` - size of the mask.
    pub fn add_segment(&mut self, mask: Array2<usize>, coord: [usize; 2], size: usize) {
        let [x, y] = coord;
        self.max_patch_id += 1;
        let patch_id = self.max_patch_id;
        let mut area = 0;
        for i in 0..size {
            for j in 0..size {
                if mask[(i, j)] == 0 {
                    continue;
                }
                let palette_x = x + i;
                let palette_y = y + j;
                let index = self.map[(palette_x, palette_y)];
                self.map[(palette_x, palette_y)] = patch_id;
                area += 1;
                if index != 0 {
                    self.areas[index - 1] -= 1;
                }
            }
        }
        self.areas.push(area);
        self.valid.push(true);
    }

    /// expand a existing segment with the given segment. Areas are responsively maintained.
    ///
    /// # Arguments:
    /// * `segment_id` - the segment id to be expanded.
    /// * `mask` - ndarray describing the mask.
    /// * `coord` - left upper corner of the mask on palette. Given in [height, width]
    /// * `size` - size of the mask.
    pub fn expand_segment(
        &mut self,
        segment_id: usize,
        mask: Array2<usize>,
        coord: [usize; 2],
        size: usize,
    ) {
        let [x, y] = coord;
        for i in 0..size {
            for j in 0..size {
                if mask[(i, j)] == 0 {
                    continue;
                }
                let palette_x = x + i;
                let palette_y = y + j;
                let index = self.map[(palette_x, palette_y)];
                self.map[(palette_x, palette_y)] = segment_id;
                if index != segment_id {
                    self.areas[segment_id - 1] += 1;
                }
                if index != 0 && index != segment_id {
                    self.areas[index - 1] -= 1;
                }
            }
        }
    }
}
