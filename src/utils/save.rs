use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct PaletteArchive {
    pub map: Array2<usize>,
    pub cluster_map: Vec<usize>,
}

impl PaletteArchive {
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let archive: PaletteArchive = serde_json::from_str(&contents)?;
        Ok(archive)
    }
}
