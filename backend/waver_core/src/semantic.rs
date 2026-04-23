use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

pub fn splade_encode_impl(
    _model_path: String,
    _texts: Vec<String>,
    _dim: usize,
    _is_query: bool,
) -> PyResult<Vec<Vec<f32>>> {
    Err(PyRuntimeError::new_err(
        "SPLADE runtime is not bundled in this build",
    ))
}

pub fn mrl_encode_impl(
    _model_path: String,
    _texts: Vec<String>,
    _dim: usize,
    _is_query: bool,
) -> PyResult<Vec<Vec<f32>>> {
    Err(PyRuntimeError::new_err(
        "MRL runtime is not bundled in this build",
    ))
}
