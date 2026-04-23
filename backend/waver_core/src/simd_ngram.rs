const SHORT_GRAM_1_SPACE: usize = 256;
const SHORT_GRAM_2_SPACE: usize = 65_536;
const TRIGRAM_SPACE: usize = 1 << 24;
const TRIGRAM_OFFSET: usize = SHORT_GRAM_1_SPACE + SHORT_GRAM_2_SPACE;
const GRAM_SPACE: usize = TRIGRAM_OFFSET + TRIGRAM_SPACE;
const BITMAP_WORDS: usize = GRAM_SPACE.div_ceil(64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelMode {
    Auto,
    #[cfg_attr(not(test), allow(dead_code))]
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

trait GramSink {
    fn push(&mut self, gram_id: u32);

    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    fn extend(&mut self, gram_ids: &[u32]) {
        for gram_id in gram_ids {
            self.push(*gram_id);
        }
    }
}

struct QueryBitmapSink<'a> {
    bitmap: &'a mut [u64],
}

impl GramSink for QueryBitmapSink<'_> {
    fn push(&mut self, gram_id: u32) {
        let (word_idx, mask) = bit_location(gram_id);
        self.bitmap[word_idx] |= mask;
    }
}

struct WindowScratch {
    seen: Vec<u64>,
    touched_words: Vec<usize>,
}

impl WindowScratch {
    fn new() -> Self {
        Self {
            seen: vec![0_u64; BITMAP_WORDS],
            touched_words: Vec::with_capacity(256),
        }
    }

    fn observe_overlap(&mut self, query_bitmap: &[u64], gram_id: u32) -> bool {
        let (word_idx, mask) = bit_location(gram_id);
        if query_bitmap[word_idx] & mask == 0 {
            return false;
        }

        let word = &mut self.seen[word_idx];
        if *word & mask != 0 {
            return false;
        }

        if *word == 0 {
            self.touched_words.push(word_idx);
        }
        *word |= mask;
        true
    }

    fn clear(&mut self) {
        for word_idx in self.touched_words.drain(..) {
            self.seen[word_idx] = 0;
        }
    }
}

struct OverlapSink<'a> {
    query_bitmap: &'a [u64],
    scratch: &'a mut WindowScratch,
    overlap: u32,
}

impl GramSink for OverlapSink<'_> {
    fn push(&mut self, gram_id: u32) {
        if self.scratch.observe_overlap(self.query_bitmap, gram_id) {
            self.overlap += 1;
        }
    }
}

pub fn prefilter_windows_impl(
    query: String,
    windows: Vec<(String, String)>,
    top_k: usize,
) -> Vec<(String, u32)> {
    prefilter_windows_with_mode(query, windows, top_k, KernelMode::Auto)
}

fn prefilter_windows_with_mode(
    query: String,
    windows: Vec<(String, String)>,
    top_k: usize,
    mode: KernelMode,
) -> Vec<(String, u32)> {
    let Some(query_bitmap) = build_query_bitmap(&query, mode) else {
        return Vec::new();
    };

    let mut scratch = WindowScratch::new();
    let mut ranked: Vec<(String, u32)> = windows
        .into_iter()
        .filter_map(|(window_id, text)| {
            let overlap = count_window_overlap(&query_bitmap, &text, &mut scratch, mode);
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

fn build_query_bitmap(query: &str, mode: KernelMode) -> Option<Vec<u64>> {
    let lowered = lowercase_bytes(query);
    if lowered.is_empty() {
        return None;
    }

    let mut bitmap = vec![0_u64; BITMAP_WORDS];
    let mut sink = QueryBitmapSink { bitmap: &mut bitmap };
    visit_grams(&lowered, &mut sink, mode);
    Some(bitmap)
}

fn count_window_overlap(
    query_bitmap: &[u64],
    text: &str,
    scratch: &mut WindowScratch,
    mode: KernelMode,
) -> u32 {
    let lowered = lowercase_bytes(text);
    if lowered.is_empty() {
        return 0;
    }

    debug_assert!(scratch.touched_words.is_empty());
    let overlap = {
        let mut sink = OverlapSink {
            query_bitmap,
            scratch,
            overlap: 0,
        };
        visit_grams(&lowered, &mut sink, mode);
        sink.overlap
    };
    scratch.clear();
    overlap
}

fn visit_grams<S: GramSink>(lowered: &[u8], sink: &mut S, mode: KernelMode) {
    match lowered.len() {
        0 => {}
        1 | 2 => sink.push(encode_short_gram(lowered).expect("short gram must encode")),
        _ => visit_trigrams(lowered, sink, mode),
    }
}

fn visit_trigrams<S: GramSink>(lowered: &[u8], sink: &mut S, mode: KernelMode) {
    debug_assert!(lowered.len() >= 3);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = mode;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::is_x86_feature_detected;

        match mode {
            KernelMode::Avx512 => {
                visit_trigrams_avx512(lowered, sink);
                return;
            }
            KernelMode::Avx2 => {
                visit_trigrams_avx2(lowered, sink);
                return;
            }
            KernelMode::Auto => {
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vl")
                {
                    visit_trigrams_avx512(lowered, sink);
                    return;
                }
                if is_x86_feature_detected!("avx2") {
                    visit_trigrams_avx2(lowered, sink);
                    return;
                }
            }
            KernelMode::Scalar => {}
        }
    }

    visit_trigrams_scalar(lowered, sink);
}

fn visit_trigrams_scalar<S: GramSink>(lowered: &[u8], sink: &mut S) {
    for start in 0..=lowered.len() - 3 {
        sink.push(encode_trigram(&lowered[start..start + 3]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn visit_trigrams_avx2<S: GramSink>(lowered: &[u8], sink: &mut S) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepu8_epi32,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_storeu_si256,
    };

    const LANES: usize = 8;
    let window_count = lowered.len() - 2;
    let mut index = 0;
    let offset = _mm256_set1_epi32(TRIGRAM_OFFSET as i32);

    while index + LANES <= window_count {
        let ptr = lowered.as_ptr().add(index);
        let left = _mm_loadl_epi64(ptr as *const __m128i);
        let middle = _mm_loadl_epi64(ptr.add(1) as *const __m128i);
        let right = _mm_loadl_epi64(ptr.add(2) as *const __m128i);

        let left32 = _mm256_cvtepu8_epi32(left);
        let middle32 = _mm256_cvtepu8_epi32(middle);
        let right32 = _mm256_cvtepu8_epi32(right);
        let merged = _mm256_add_epi32(
            _mm256_or_si256(
                _mm256_or_si256(_mm256_slli_epi32(left32, 16), _mm256_slli_epi32(middle32, 8)),
                right32,
            ),
            offset,
        );

        let mut buffer = [0_u32; LANES];
        _mm256_storeu_si256(buffer.as_mut_ptr() as *mut __m256i, merged);
        sink.extend(&buffer);
        index += LANES;
    }

    for start in index..window_count {
        sink.push(encode_trigram(&lowered[start..start + 3]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "avx512vl")]
unsafe fn visit_trigrams_avx512<S: GramSink>(lowered: &[u8], sink: &mut S) {
    use std::arch::x86_64::{
        __m128i, __m512i, _mm_loadu_si128, _mm512_add_epi32, _mm512_cvtepu8_epi32,
        _mm512_or_si512, _mm512_set1_epi32, _mm512_slli_epi32, _mm512_storeu_si512,
    };

    const LANES: usize = 16;
    let window_count = lowered.len() - 2;
    let mut index = 0;
    let offset = _mm512_set1_epi32(TRIGRAM_OFFSET as i32);

    while index + LANES <= window_count {
        let ptr = lowered.as_ptr().add(index);
        let left = _mm_loadu_si128(ptr as *const __m128i);
        let middle = _mm_loadu_si128(ptr.add(1) as *const __m128i);
        let right = _mm_loadu_si128(ptr.add(2) as *const __m128i);

        let left32 = _mm512_cvtepu8_epi32(left);
        let middle32 = _mm512_cvtepu8_epi32(middle);
        let right32 = _mm512_cvtepu8_epi32(right);
        let merged = _mm512_add_epi32(
            _mm512_or_si512(
                _mm512_or_si512(_mm512_slli_epi32(left32, 16), _mm512_slli_epi32(middle32, 8)),
                right32,
            ),
            offset,
        );

        let mut buffer = [0_u32; LANES];
        _mm512_storeu_si512(buffer.as_mut_ptr() as *mut __m512i, merged);
        sink.extend(&buffer);
        index += LANES;
    }

    for start in index..window_count {
        sink.push(encode_trigram(&lowered[start..start + 3]));
    }
}

fn lowercase_bytes(text: &str) -> Vec<u8> {
    text.to_lowercase().into_bytes()
}

fn encode_short_gram(bytes: &[u8]) -> Option<u32> {
    match bytes.len() {
        1 => Some(bytes[0] as u32),
        2 => Some((SHORT_GRAM_1_SPACE as u32) + ((bytes[0] as u32) << 8) + (bytes[1] as u32)),
        _ => None,
    }
}

fn encode_trigram(bytes: &[u8]) -> u32 {
    debug_assert_eq!(bytes.len(), 3);
    (TRIGRAM_OFFSET as u32)
        + ((bytes[0] as u32) << 16)
        + ((bytes[1] as u32) << 8)
        + (bytes[2] as u32)
}

fn bit_location(gram_id: u32) -> (usize, u64) {
    let bit_index = gram_id as usize;
    (bit_index / 64, 1_u64 << (bit_index % 64))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{prefilter_windows_with_mode, KernelMode};

    fn reference_byte_ngrams(text: &str, size: usize) -> HashSet<Vec<u8>> {
        let raw = text.to_lowercase().into_bytes();
        if raw.is_empty() {
            return HashSet::new();
        }
        if raw.len() < size {
            return HashSet::from([raw]);
        }

        let mut grams = HashSet::new();
        for index in 0..=(raw.len() - size) {
            grams.insert(raw[index..index + size].to_vec());
        }
        grams
    }

    fn reference_prefilter(
        query: &str,
        windows: Vec<(String, String)>,
        top_k: usize,
    ) -> Vec<(String, u32)> {
        let query_grams = reference_byte_ngrams(query, 3);
        if query_grams.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<(String, u32)> = windows
            .into_iter()
            .filter_map(|(window_id, text)| {
                let grams = reference_byte_ngrams(&text, 3);
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

    #[test]
    fn test_duplicate_grams_do_not_double_count() {
        let result = prefilter_windows_with_mode(
            "aaaa".to_string(),
            vec![("w-1".to_string(), "aaaaaa".to_string())],
            5,
            KernelMode::Scalar,
        );

        assert_eq!(result, vec![("w-1".to_string(), 1)]);
    }

    #[test]
    fn test_short_strings_and_empty_strings_preserve_overlap() {
        let windows = vec![
            ("a".to_string(), "".to_string()),
            ("b".to_string(), "A".to_string()),
            ("c".to_string(), "ab".to_string()),
            ("d".to_string(), "abc".to_string()),
        ];

        let short = prefilter_windows_with_mode(
            "a".to_string(),
            windows.clone(),
            10,
            KernelMode::Scalar,
        );
        let pair = prefilter_windows_with_mode(
            "ab".to_string(),
            windows.clone(),
            10,
            KernelMode::Scalar,
        );
        let tri = prefilter_windows_with_mode(
            "abc".to_string(),
            windows,
            10,
            KernelMode::Scalar,
        );

        assert_eq!(short, vec![("b".to_string(), 1)]);
        assert_eq!(pair, vec![("c".to_string(), 1)]);
        assert_eq!(tri, vec![("d".to_string(), 1)]);
    }

    #[test]
    fn test_unicode_lowercasing_uses_utf8_bytes_exactly() {
        let result = prefilter_windows_with_mode(
            "Å".to_string(),
            vec![
                ("hit".to_string(), "å".to_string()),
                ("miss".to_string(), "a".to_string()),
            ],
            5,
            KernelMode::Scalar,
        );

        assert_eq!(result, vec![("hit".to_string(), 1)]);
    }

    #[test]
    fn test_tie_ordering_and_top_k_truncation_match_contract() {
        let result = prefilter_windows_with_mode(
            "fatal timeout marker".to_string(),
            vec![
                ("b".to_string(), "fatal timeout marker".to_string()),
                ("a".to_string(), "fatal timeout marker".to_string()),
                ("c".to_string(), "fatal timeout".to_string()),
            ],
            2,
            KernelMode::Scalar,
        );

        assert_eq!(
            result,
            vec![
                ("a".to_string(), 18),
                ("b".to_string(), 18),
            ]
        );
    }

    #[test]
    fn test_scalar_path_matches_reference_prefilter() {
        let windows = vec![
            ("w-1".to_string(), "heartbeat heartbeat heartbeat".to_string()),
            ("w-2".to_string(), "fatal timeout marker 9f2 observed in prod".to_string()),
            ("w-3".to_string(), "gateway timeout with error code 503".to_string()),
            ("w-4".to_string(), "shipping notice and unrelated update".to_string()),
            ("w-5".to_string(), "Å resolved in cafÉ after retry".to_string()),
            ("w-6".to_string(), "ab".to_string()),
        ];

        let actual = prefilter_windows_with_mode(
            "fatal timeout marker 9f2".to_string(),
            windows.clone(),
            4,
            KernelMode::Scalar,
        );
        let expected = reference_prefilter("fatal timeout marker 9f2", windows, 4);

        assert_eq!(actual, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_path_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let windows = vec![
            ("w-1".to_string(), "fatal timeout marker 9f2 observed in prod".to_string()),
            ("w-2".to_string(), "fatal timeout marker 9f2 observed in staging".to_string()),
            ("w-3".to_string(), "shipping notice and unrelated update".to_string()),
        ];

        let actual = prefilter_windows_with_mode(
            "fatal timeout marker 9f2".to_string(),
            windows.clone(),
            3,
            KernelMode::Avx2,
        );
        let expected = reference_prefilter("fatal timeout marker 9f2", windows, 3);

        assert_eq!(actual, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_path_matches_reference_when_available() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }

        let windows = vec![
            ("w-1".to_string(), "fatal timeout marker 9f2 observed in prod".to_string()),
            ("w-2".to_string(), "fatal timeout marker 9f2 observed in staging".to_string()),
            ("w-3".to_string(), "gateway timeout with error code 503".to_string()),
        ];

        let actual = prefilter_windows_with_mode(
            "fatal timeout marker 9f2".to_string(),
            windows.clone(),
            3,
            KernelMode::Avx512,
        );
        let expected = reference_prefilter("fatal timeout marker 9f2", windows, 3);

        assert_eq!(actual, expected);
    }
}
