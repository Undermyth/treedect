use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};

use crate::panels::palette;

#[derive(Clone)]
pub struct TableEntry {
    pub id: usize,
    pub score: f32,
    pub number_score: f32,
    pub area_score: f32,
    pub grid_score: f32,
}

impl TableEntry {
    pub fn default() -> Self {
        Self {
            id: 0,
            score: 0.0,
            number_score: 0.0,
            area_score: 0.0,
            grid_score: 0.0,
        }
    }
}

pub struct ColumnConfig {
    pub header: String,
    pub value_extractor: Box<dyn Fn(&TableEntry) -> String>,
}

impl ColumnConfig {
    pub fn new(header: String, value_extractor: Box<dyn Fn(&TableEntry) -> String>) -> Self {
        Self {
            header,
            value_extractor,
        }
    }
}

pub struct Table {
    pub entries: Vec<TableEntry>,
}

impl Table {
    pub fn new() -> Self {
        Self {
            entries: Vec::<TableEntry>::new(),
        }
    }

    fn get_default_columns() -> Vec<ColumnConfig> {
        vec![
            ColumnConfig::new("ID".to_string(), Box::new(|entry| entry.id.to_string())),
            ColumnConfig::new(
                "Score".to_string(),
                Box::new(|entry| entry.score.to_string()),
            ),
            ColumnConfig::new(
                "Number Score".to_string(),
                Box::new(|entry| entry.number_score.to_string()),
            ),
            ColumnConfig::new(
                "Area Score".to_string(),
                Box::new(|entry| entry.area_score.to_string()),
            ),
            ColumnConfig::new(
                "Grid Score".to_string(),
                Box::new(|entry| entry.grid_score.to_string()),
            ),
        ]
    }

    pub fn build_from_palette(palette: Arc<Mutex<palette::Palette>>) -> Self {
        let palette = palette.lock().unwrap();
        let mut table = Self::new();
        table
            .entries
            .resize(palette.num_clusters, TableEntry::default());
        for (index, entry) in table.entries.iter_mut().enumerate() {
            entry.id = index + 1;
        }
        let mut total_number = 0;
        let mut total_area = 0;
        let mut total_grid = 0;
        for (index, cluster_id) in palette.cluster_map.iter().enumerate() {
            // index start from 0, cluster id start from 1
            if *cluster_id == 0 {
                continue;
            }
            let area = palette.areas[index];
            let grid = palette.grids[index];
            total_number += 1;
            total_area += area;
            total_grid += grid;
            table.entries[*cluster_id - 1].number_score += 1 as f32;
            table.entries[*cluster_id - 1].area_score += area as f32;
            table.entries[*cluster_id - 1].grid_score += grid as f32;
        }
        for entry in table.entries.iter_mut() {
            entry.area_score /= total_area as f32;
            entry.grid_score /= total_grid as f32;
            entry.number_score /= total_number as f32;
            entry.score = (entry.number_score + entry.area_score + entry.grid_score) / 3.0;
        }
        table
    }
    pub fn sort_by_score(&mut self, descending: bool) {
        if descending {
            self.entries
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            self.entries
                .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }
    }
    pub fn sort_by_id(&mut self, descending: bool) {
        if descending {
            self.entries
                .sort_by(|a, b| b.id.partial_cmp(&a.id).unwrap());
        } else {
            self.entries
                .sort_by(|a, b| a.id.partial_cmp(&b.id).unwrap());
        }
    }
    pub fn export_to_csv(&self, path: String) -> std::io::Result<()> {
        let columns = Self::get_default_columns();
        let mut file = File::create(path)?;

        // Write header row
        let header: Vec<String> = columns.iter().map(|c| c.header.clone()).collect();
        writeln!(file, "{}", header.join(","))?;

        // Write data rows
        for entry in &self.entries {
            let row: Vec<String> = columns.iter().map(|c| (c.value_extractor)(entry)).collect();
            writeln!(file, "{}", row.join(","))?;
        }

        Ok(())
    }
}
