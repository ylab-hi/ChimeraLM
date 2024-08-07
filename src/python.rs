use deepbiop::utils;
use log::{debug, error, info, warn};
use pyo3::prelude::*;

#[pyfunction]
fn test_deepbiop() {
    let segment = utils::interval::Segment::new("chr2", 100, 2000).unwrap();
    println!("{:?}", segment);
}

#[pyfunction]
fn test_log() {
    debug!("debug Hello from Rust!");
    info!("info Hello from Rust!");
    warn!("warn Hello from Rust!");
    error!("error Hello from Rust!");
}

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}
/// A Python module implemented in Rust.
#[pymodule]
fn chimera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_log, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(test_deepbiop, m)?)?;
    Ok(())
}
