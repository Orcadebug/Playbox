# waver_core

Stage-1 Rust core for retrieval primitives.

## Build and install locally

```bash
cd backend/waver_core
maturin develop --release
```

After installation, the backend can call `waver_core.rrf_fuse` when
`WAVER_RUST_RRF=true`.
