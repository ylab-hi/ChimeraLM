use std::io::Write;
use std::path::PathBuf;

use ahash::HashMap;
use ahash::HashMapExt;
use anyhow::Result;
use candle_core::{self, pickle};
use log::info;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use walkdir::WalkDir;

#[pyclass]
#[derive(Debug, Default, FromPyObject, Deserialize, Serialize)]
pub struct Predict {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub label: i64,

    #[pyo3(get, set)]
    pub sv: Option<String>,
}

// implement repr for Predict
impl std::fmt::Display for Predict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "id: {}, label: {}", self.id, self.label)
    }
}

pub fn ascii_list2str(ascii_list: &[i64]) -> String {
    ascii_list
        .par_iter()
        .map(|&c| char::from_u32(c as u32).unwrap())
        .collect()
}

#[pyfunction]
pub fn write_predicts_to_file(predicts: HashMap<String, Predict>, pt_path: PathBuf) -> Result<()> {
    // write predicts to a txt  file
    // column 1: id
    // column 2: label

    let file = std::fs::File::create(pt_path)?;
    let mut writer = std::io::BufWriter::new(file);

    for (_, predict) in predicts {
        writeln!(writer, "{},{}", predict.id, predict.label)?;
    }

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (pt_path, max_predicts = None))]
pub fn load_predicts_from_batch_pts(
    pt_path: PathBuf,
    max_predicts: Option<usize>,
) -> Result<HashMap<String, Predict>> {
    // iter over the pt files under the path
    // makes sure there is only one pt file
    let mut pt_files: Vec<_> = WalkDir::new(&pt_path)
        .into_iter()
        .par_bridge()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "pt"))
        .collect();

    info!(
        "Found {} pt files from {}",
        pt_files.len(),
        pt_path.display()
    );

    if let Some(max_predicts) = max_predicts {
        if pt_files.len() > max_predicts {
            info!("only load first {max_predicts} pt files");
            pt_files.truncate(max_predicts);
        }
    }

    // Use Rayon to process files in parallel
    let result: Result<Vec<_>> = pt_files
        .into_par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            match load_predicts_from_batch_pt(path.to_path_buf()) {
                Ok(predicts) => Some(Ok(predicts)),
                Err(e) => {
                    println!(
                        "load pt {} fail caused by Error: {:?}",
                        path.to_string_lossy(),
                        e
                    );
                    None
                }
            }
        })
        .collect();
    result.map(|vectors| vectors.into_par_iter().flatten().collect())
}

#[pyfunction]
pub fn load_predicts_from_batch_pt(pt_path: PathBuf) -> Result<HashMap<String, Predict>> {
    let tensors = pickle::read_all(pt_path).unwrap();
    let mut tensors_map = HashMap::new();

    for (key, value) in tensors {
        tensors_map.insert(key, value);
    }

    let _predictions = tensors_map.get("prediction").unwrap().argmax(1)?; // shape batch, seq_len
    let predictions = _predictions.to_dtype(candle_core::DType::I64)?;
    let predictions_vec = predictions.to_vec1::<i64>()?;

    let id = tensors_map.get("id").unwrap();
    let id_vec = id.to_vec2::<i64>()?;

    let batch_size = predictions_vec.len();

    Ok((0..batch_size)
        .into_par_iter()
        .map(|i| {
            let id_data_end = id_vec[i][0] as usize + 1;
            let id_data = &id_vec[i][1..id_data_end];

            let id = ascii_list2str(id_data);
            let label = predictions_vec[i];
            (
                id.clone(),
                Predict {
                    id,
                    label,
                    sv: None,
                },
            )
        })
        .collect::<HashMap<_, _>>())
}
