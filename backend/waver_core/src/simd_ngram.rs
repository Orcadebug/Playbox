use std::collections::HashSet;

fn byte_ngrams(text: &str, size: usize) -> HashSet<Vec<u8>> {
    let raw = text.to_lowercase().into_bytes();
    if raw.is_empty() {
        return HashSet::new();
    }
    if raw.len() < size {
        return HashSet::from([raw]);
    }

    let mut grams: HashSet<Vec<u8>> = HashSet::new();
    for index in 0..=(raw.len() - size) {
        grams.insert(raw[index..index + size].to_vec());
    }
    grams
}

pub fn prefilter_windows_impl(
    query: String,
    windows: Vec<(String, String)>,
    top_k: usize,
) -> Vec<(String, u32)> {
    let query_grams = byte_ngrams(&query, 3);
    if query_grams.is_empty() {
        return Vec::new();
    }

    let mut ranked: Vec<(String, u32)> = windows
        .into_iter()
        .filter_map(|(window_id, text)| {
            let grams = byte_ngrams(&text, 3);
            let overlap = query_grams.intersection(&grams).count() as u32;
            if overlap == 0 {
                None
            } else {
                Some((window_id, overlap))
            }
        })
        .collect();

    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    if ranked.len() > top_k {
        ranked.truncate(top_k);
    }
    ranked
}
