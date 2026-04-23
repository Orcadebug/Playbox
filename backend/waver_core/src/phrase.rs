use aho_corasick::AhoCorasick;

pub fn phrase_search_impl(
    phrases: Vec<String>,
    haystacks: Vec<(String, String)>,
    top_k: usize,
) -> Vec<(String, f32)> {
    if phrases.is_empty() || haystacks.is_empty() {
        return Vec::new();
    }

    let automaton = match AhoCorasick::new(phrases) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };

    let mut hits: Vec<(String, f32)> = haystacks
        .into_iter()
        .filter_map(|(chunk_id, haystack)| {
            if automaton.is_match(&haystack) {
                Some((chunk_id, 1.0))
            } else {
                None
            }
        })
        .collect();
    hits.sort_by(|left, right| left.0.cmp(&right.0));
    if hits.len() > top_k {
        hits.truncate(top_k);
    }
    hits
}
