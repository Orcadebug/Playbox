use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, TantivyDocument, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy};

#[pyclass]
pub struct RustBm25Index {
    index: Index,
    reader: IndexReader,
    #[allow(dead_code)]
    writer: IndexWriter,
    chunk_id_field: tantivy::schema::Field,
    body_field: tantivy::schema::Field,
}

#[pymethods]
impl RustBm25Index {
    #[new]
    fn new(documents: Vec<(String, String)>) -> PyResult<Self> {
        let mut schema_builder = Schema::builder();
        let chunk_id_field = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let body_field = schema_builder.add_text_field("body", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index
            .writer(50_000_000)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        for (chunk_id, body) in documents {
            writer
                .add_document(doc!(chunk_id_field => chunk_id, body_field => body))
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        }
        writer
            .commit()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        reader
            .reload()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        Ok(Self {
            index,
            reader,
            writer,
            chunk_id_field,
            body_field,
        })
    }

    fn search(&self, query: String, top_k: usize) -> PyResult<Vec<(String, f32)>> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        let parsed_query = query_parser
            .parse_query(&query)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(top_k))
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let mut rows: Vec<(String, f32)> = Vec::with_capacity(top_docs.len());
        for (score, address) in top_docs {
            let document: TantivyDocument = searcher
                .doc(address)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let chunk_id: &str = document
                .get_first(self.chunk_id_field)
                .and_then(|value| value.as_str())
                .ok_or_else(|| PyRuntimeError::new_err("chunk_id missing from tantivy document"))?;
            rows.push((chunk_id.to_string(), score));
        }
        rows.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        Ok(rows)
    }
}
