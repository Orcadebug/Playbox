use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array2, ArrayViewD, Axis};
use ort::session::Session;
use ort::value::{Outlet, TensorElementType, TensorRef};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokenizers::tokenizer::{Encoding, PaddingParams};
use tokenizers::utils::truncation::{
    TruncationDirection, TruncationParams, TruncationStrategy,
};
use tokenizers::Tokenizer;

const DEFAULT_MAX_LENGTH: usize = 512;
const ORT_ENV_VAR: &str = "ORT_DYLIB_PATH";
const WAVER_ORT_ENV_VAR: &str = "WAVER_ORT_DYLIB_PATH";

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static MRL_RUNTIME_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<MrlRuntime>>>> = OnceLock::new();

#[derive(Clone, Debug)]
struct InputNames {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PoolingMode {
    Direct,
    MeanPool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputSelector {
    output_index: usize,
    pooling_mode: PoolingMode,
}

struct TokenBatch {
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    token_type_ids: Array2<i64>,
}

struct MrlRuntime {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    input_names: InputNames,
    output_selector: OutputSelector,
}

impl MrlRuntime {
    fn load(model_path: &Path) -> PyResult<Self> {
        ensure_ort_initialized()?;

        let tokenizer_path = model_path
            .parent()
            .map(|parent| parent.join("tokenizer.json"))
            .ok_or_else(|| PyRuntimeError::new_err("MRL model path has no parent directory"))?;
        let tokenizer = load_tokenizer(&tokenizer_path)?;

        let session = Session::builder()
            .map_err(to_runtime_error)?
            .commit_from_file(model_path)
            .map_err(to_runtime_error)?;
        let input_names = discover_input_names(session.inputs())?;
        let output_selector = select_output(session.outputs())?;

        Ok(Self {
            tokenizer,
            session: Mutex::new(session),
            input_names,
            output_selector,
        })
    }

    fn encode(&self, texts: &[String], dim: usize) -> PyResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch = encode_texts(&self.tokenizer, texts)?;
        let input_ids = TensorRef::from_array_view(batch.input_ids.view()).map_err(to_runtime_error)?;
        let attention_mask =
            TensorRef::from_array_view(batch.attention_mask.view()).map_err(to_runtime_error)?;
        let token_type_ids =
            TensorRef::from_array_view(batch.token_type_ids.view()).map_err(to_runtime_error)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| PyRuntimeError::new_err("MRL session mutex poisoned"))?;

        let mut inputs = ort::inputs! {
            self.input_names.input_ids.as_str() => input_ids,
            self.input_names.attention_mask.as_str() => attention_mask,
        };
        if let Some(token_type_name) = &self.input_names.token_type_ids {
            inputs.push((
                Cow::Owned(token_type_name.clone()),
                ort::session::SessionInputValue::from(token_type_ids),
            ));
        }

        let outputs = session.run(inputs).map_err(to_runtime_error)?;
        let output = &outputs[self.output_selector.output_index];
        let embeddings = output.try_extract_array::<f32>().map_err(to_runtime_error)?;

        extract_embeddings(
            embeddings,
            &batch.attention_mask,
            dim,
            self.output_selector.pooling_mode,
        )
    }
}

pub fn mrl_encode_impl(
    model_path: String,
    texts: Vec<String>,
    dim: usize,
    is_query: bool,
) -> PyResult<Vec<Vec<f32>>> {
    let _ = is_query;
    let raw_model_path = PathBuf::from(model_path);
    let model_path = fs::canonicalize(&raw_model_path).unwrap_or(raw_model_path);
    let runtime = cached_runtime(&model_path)?;
    runtime.encode(&texts, dim)
}

fn cached_runtime(model_path: &Path) -> PyResult<Arc<MrlRuntime>> {
    let cache = MRL_RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(runtime) = cache
        .lock()
        .map_err(|_| PyRuntimeError::new_err("MRL runtime cache mutex poisoned"))?
        .get(model_path)
        .cloned()
    {
        return Ok(runtime);
    }

    let runtime = Arc::new(MrlRuntime::load(model_path)?);
    let mut guard = cache
        .lock()
        .map_err(|_| PyRuntimeError::new_err("MRL runtime cache mutex poisoned"))?;
    let entry = guard
        .entry(model_path.to_path_buf())
        .or_insert_with(|| Arc::clone(&runtime));
    Ok(Arc::clone(entry))
}

fn ensure_ort_initialized() -> PyResult<()> {
    match ORT_INIT.get_or_init(init_ort_runtime) {
        Ok(()) => Ok(()),
        Err(message) => Err(PyRuntimeError::new_err(message.clone())),
    }
}

fn init_ort_runtime() -> Result<(), String> {
    let dylib_path = discover_ort_dylib_path()
        .ok_or_else(|| format!("Unable to locate ONNX Runtime dylib; set {WAVER_ORT_ENV_VAR} or {ORT_ENV_VAR}, or install Python onnxruntime"))?;
    let _ = ort::environment::init_from(&dylib_path)
        .map_err(|err| format!("Failed to load ONNX Runtime from {}: {err}", dylib_path.display()))?
        .with_name("waver-mrl")
        .commit();
    Ok(())
}

fn discover_ort_dylib_path() -> Option<PathBuf> {
    for key in [WAVER_ORT_ENV_VAR, ORT_ENV_VAR] {
        if let Some(path) = env::var_os(key) {
            let candidate = PathBuf::from(path);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    Python::with_gil(|py| {
        let module = PyModule::import_bound(py, "onnxruntime").ok()?;
        let module_file: String = module.getattr("__file__").ok()?.extract().ok()?;
        let package_root = PathBuf::from(module_file).parent()?.to_path_buf();
        find_ort_dylib(&package_root)
    })
}

fn find_ort_dylib(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut matches = Vec::new();

    while let Some(path) = stack.pop() {
        let entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let candidate = entry.path();
            if candidate.is_dir() {
                stack.push(candidate);
                continue;
            }
            if is_ort_dylib(&candidate) {
                matches.push(candidate);
            }
        }
    }

    matches.sort();
    matches.into_iter().next()
}

fn is_ort_dylib(path: &Path) -> bool {
    let file_name = match path.file_name().and_then(|value| value.to_str()) {
        Some(value) => value,
        None => return false,
    };

    #[cfg(target_os = "macos")]
    {
        file_name.starts_with("libonnxruntime") && file_name.ends_with(".dylib")
    }
    #[cfg(target_os = "windows")]
    {
        file_name.starts_with("onnxruntime") && file_name.ends_with(".dll")
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        file_name.starts_with("libonnxruntime") && file_name.contains(".so")
    }
}

fn load_tokenizer(tokenizer_path: &Path) -> PyResult<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to load tokenizer.json: {err}")))?;

    let padding = tokenizer.get_padding().cloned().unwrap_or_else(|| PaddingParams {
        pad_id: tokenizer.token_to_id("[PAD]").unwrap_or(0),
        pad_token: "[PAD]".to_string(),
        ..Default::default()
    });
    tokenizer.with_padding(Some(padding));
    tokenizer
        .with_truncation(Some(TruncationParams {
            direction: TruncationDirection::Right,
            max_length: DEFAULT_MAX_LENGTH,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .map_err(|err| PyRuntimeError::new_err(format!("Failed to configure tokenizer truncation: {err}")))?;
    Ok(tokenizer)
}

fn discover_input_names(inputs: &[Outlet]) -> PyResult<InputNames> {
    let input_ids = find_input_name(inputs, &["input_ids"])?;
    let attention_mask = find_input_name(inputs, &["attention_mask"])?;
    let token_type_ids = inputs
        .iter()
        .find(|input| input.name() == "token_type_ids")
        .map(|input| input.name().to_string());
    Ok(InputNames {
        input_ids,
        attention_mask,
        token_type_ids,
    })
}

fn find_input_name(inputs: &[Outlet], candidates: &[&str]) -> PyResult<String> {
    for candidate in candidates {
        if let Some(input) = inputs.iter().find(|input| input.name() == *candidate) {
            return Ok(input.name().to_string());
        }
    }
    Err(PyRuntimeError::new_err(format!(
        "MRL model is missing required input(s): {}",
        candidates.join(", ")
    )))
}

fn select_output(outputs: &[Outlet]) -> PyResult<OutputSelector> {
    let direct_names = ["sentence_embedding", "pooled_output", "pooler_output"];
    if let Some(output_index) = select_output_by_name(outputs, &direct_names, 2) {
        return Ok(OutputSelector {
            output_index,
            pooling_mode: PoolingMode::Direct,
        });
    }

    if let Some(output_index) = outputs
        .iter()
        .position(|output| outlet_is_float_tensor_rank(output, 2))
    {
        return Ok(OutputSelector {
            output_index,
            pooling_mode: PoolingMode::Direct,
        });
    }

    let token_names = ["last_hidden_state", "token_embeddings"];
    if let Some(output_index) = select_output_by_name(outputs, &token_names, 3) {
        return Ok(OutputSelector {
            output_index,
            pooling_mode: PoolingMode::MeanPool,
        });
    }

    if let Some(output_index) = outputs
        .iter()
        .position(|output| outlet_is_float_tensor_rank(output, 3))
    {
        return Ok(OutputSelector {
            output_index,
            pooling_mode: PoolingMode::MeanPool,
        });
    }

    Err(PyRuntimeError::new_err(
        "MRL model does not expose a supported float tensor output",
    ))
}

fn select_output_by_name(outputs: &[Outlet], names: &[&str], dimensions: usize) -> Option<usize> {
    outputs.iter().position(|output| {
        outlet_is_float_tensor_rank(output, dimensions)
            && names
                .iter()
                .any(|needle| output.name().eq_ignore_ascii_case(needle))
    })
}

fn outlet_is_float_tensor_rank(output: &Outlet, rank: usize) -> bool {
    output.dtype().tensor_type() == Some(TensorElementType::Float32)
        && output
            .dtype()
            .tensor_shape()
            .is_some_and(|shape| shape.len() == rank)
}

fn encode_texts(tokenizer: &Tokenizer, texts: &[String]) -> PyResult<TokenBatch> {
    let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|err| PyRuntimeError::new_err(format!("MRL tokenization failed: {err}")))?;
    let seq_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(0);
    let pad_id = tokenizer
        .get_padding()
        .map(|padding| i64::from(padding.pad_id))
        .unwrap_or(0);

    Ok(TokenBatch {
        input_ids: build_i64_matrix_with_default(&encodings, seq_len, Encoding::get_ids, pad_id),
        attention_mask: build_i64_matrix_with_default(
            &encodings,
            seq_len,
            Encoding::get_attention_mask,
            0,
        ),
        token_type_ids: build_i64_matrix_with_default(
            &encodings,
            seq_len,
            Encoding::get_type_ids,
            0,
        ),
    })
}

fn build_i64_matrix_with_default(
    encodings: &[Encoding],
    seq_len: usize,
    getter: fn(&Encoding) -> &[u32],
    default: i64,
) -> Array2<i64> {
    Array2::from_shape_fn((encodings.len(), seq_len), |(row, col)| {
        getter(&encodings[row])
            .get(col)
            .map_or(default, |value| *value as i64)
    })
}

fn extract_embeddings(
    embeddings: ArrayViewD<'_, f32>,
    attention_mask: &Array2<i64>,
    dim: usize,
    pooling_mode: PoolingMode,
) -> PyResult<Vec<Vec<f32>>> {
    match pooling_mode {
        PoolingMode::Direct => extract_direct_embeddings(embeddings, dim),
        PoolingMode::MeanPool => extract_mean_pooled_embeddings(embeddings, attention_mask, dim),
    }
}

fn extract_direct_embeddings(embeddings: ArrayViewD<'_, f32>, dim: usize) -> PyResult<Vec<Vec<f32>>> {
    let matrix = embeddings
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| PyRuntimeError::new_err("Expected MRL model output shape [batch, hidden]"))?;
    if matrix.ncols() < dim {
        return Err(PyRuntimeError::new_err(format!(
            "MRL model output dimension {} is smaller than requested dim {}",
            matrix.ncols(),
            dim
        )));
    }

    Ok(matrix
        .axis_iter(Axis(0))
        .map(|row| row.iter().take(dim).copied().collect())
        .collect())
}

fn extract_mean_pooled_embeddings(
    embeddings: ArrayViewD<'_, f32>,
    attention_mask: &Array2<i64>,
    dim: usize,
) -> PyResult<Vec<Vec<f32>>> {
    let tensor = embeddings
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| PyRuntimeError::new_err("Expected MRL model output shape [batch, seq, hidden]"))?;
    if tensor.shape()[2] < dim {
        return Err(PyRuntimeError::new_err(format!(
            "MRL model output dimension {} is smaller than requested dim {}",
            tensor.shape()[2],
            dim
        )));
    }

    let mut rows = Vec::with_capacity(tensor.shape()[0]);
    for (row_index, row) in tensor.axis_iter(Axis(0)).enumerate() {
        let mut pooled = vec![0.0_f32; dim];
        let mut weight_sum = 0.0_f32;
        let seq_len = row.shape()[0].min(attention_mask.ncols());

        for token_index in 0..seq_len {
            if attention_mask[[row_index, token_index]] <= 0 {
                continue;
            }
            weight_sum += 1.0;
            let token = row.index_axis(Axis(0), token_index);
            for (slot, value) in pooled.iter_mut().zip(token.iter().take(dim)) {
                *slot += *value;
            }
        }

        if weight_sum > 0.0 {
            for value in &mut pooled {
                *value /= weight_sum;
            }
        }
        rows.push(pooled);
    }

    Ok(rows)
}

fn to_runtime_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
