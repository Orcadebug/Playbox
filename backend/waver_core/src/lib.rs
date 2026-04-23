use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

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

#[pymodule]
fn waver_core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rrf_fuse, module)?)?;
    Ok(())
}
