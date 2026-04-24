mod bm25;
mod mrl;
mod phrase;
mod semantic;
mod simd_ngram;

use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::bm25::RustBm25Index;
use crate::mrl::mrl_encode_impl;
use crate::phrase::phrase_search_impl;
use crate::semantic::splade_encode_impl;
use crate::simd_ngram::{prefilter_windows_impl, visit_trigrams_avx512_impl};

type HeadPayload = Vec<(String, Vec<(String, f64, Option<f64>)>)>;
type FusedRow = (String, f64, Vec<String>, HashMap<String, f64>, Option<f64>);

#[derive(Default)]
struct Slot {
    score: f64,
    channels: Vec<String>,
    channel_scores: HashMap<String, f64>,
    bm25_score: Option<f64>,
}

#[pyfunction]
fn rrf_fuse(head_results: HeadPayload, top_k: usize, rrf_k: usize) -> Vec<FusedRow> {
    let mut fused: HashMap<String, Slot> = HashMap::new();

    for (head_name, hits) in head_results {
        for (rank, (chunk_id, score, bm25_score)) in hits.into_iter().enumerate() {
            let slot = fused.entry(chunk_id).or_default();
            slot.score += 1.0 / ((rrf_k + rank + 1) as f64);
            slot.channels.push(head_name.clone());
            slot.channel_scores.insert(head_name.clone(), score);
            if slot.bm25_score.is_none() && bm25_score.is_some() {
                slot.bm25_score = bm25_score;
            }
        }
    }

    let mut rows: Vec<FusedRow> = fused
        .into_iter()
        .map(|(chunk_id, slot)| {
            (
                chunk_id,
                slot.score,
                slot.channels,
                slot.channel_scores,
                slot.bm25_score,
            )
        })
        .collect();

    rows.sort_by(|a, b| {
        let score_cmp = b.1.total_cmp(&a.1);
        if score_cmp == Ordering::Equal {
            a.0.cmp(&b.0)
        } else {
            score_cmp
        }
    });

    if rows.len() > top_k {
        rows.truncate(top_k);
    }
    rows
}

#[pyfunction]
fn phrase_search(
    phrases: Vec<String>,
    haystacks: Vec<(String, String)>,
    top_k: usize,
) -> Vec<(String, f32)> {
    phrase_search_impl(phrases, haystacks, top_k)
}

#[pyfunction]
fn prefilter_windows(
    query: String,
    windows: Vec<(String, String)>,
    top_k: usize,
) -> Vec<(String, u32)> {
    prefilter_windows_impl(query, windows, top_k)
}

#[pyfunction]
fn extract_direct_embeddings(rows: Vec<Vec<f32>>, dim: usize) -> PyResult<Vec<Vec<f32>>> {
    rows.into_iter()
        .map(|row| {
            if row.len() < dim {
                Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "MRL model output dimension {} is smaller than requested dim {}",
                    row.len(),
                    dim
                )))
            } else {
                Ok(row.into_iter().take(dim).collect())
            }
        })
        .collect()
}

#[pyfunction]
fn visit_trigrams_avx512(text: String) -> PyResult<Vec<u32>> {
    visit_trigrams_avx512_impl(text).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
fn splade_encode(
    model_path: String,
    texts: Vec<String>,
    dim: usize,
    is_query: bool,
) -> PyResult<Vec<Vec<f32>>> {
    splade_encode_impl(model_path, texts, dim, is_query)
}

#[pyfunction]
fn mrl_encode(
    model_path: String,
    texts: Vec<String>,
    dim: usize,
    is_query: bool,
) -> PyResult<Vec<Vec<f32>>> {
    mrl_encode_impl(model_path, texts, dim, is_query)
}

#[pymodule]
fn waver_core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustBm25Index>()?;
    module.add_function(wrap_pyfunction!(rrf_fuse, module)?)?;
    module.add_function(wrap_pyfunction!(phrase_search, module)?)?;
    module.add_function(wrap_pyfunction!(prefilter_windows, module)?)?;
    module.add_function(wrap_pyfunction!(extract_direct_embeddings, module)?)?;
    module.add_function(wrap_pyfunction!(visit_trigrams_avx512, module)?)?;
    module.add_function(wrap_pyfunction!(splade_encode, module)?)?;
    module.add_function(wrap_pyfunction!(mrl_encode, module)?)?;
    Ok(())
}
