use pyo3::prelude::*;

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    println!("Adding {a} + {b}");
    a + b
}

/// A Python module implemented in Rust.
#[pymodule]
fn chimera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
