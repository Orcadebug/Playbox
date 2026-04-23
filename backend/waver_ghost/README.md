# Waver Ghost

Optional edge-side short-circuit service for chunked uploads.

It keeps lightweight Bloom and Count-Min style summaries so clients that opt
into chunked upload can cheaply detect obvious zero-hit queries before routing
the request into the main retrieval stack. Non-chunked clients bypass it.
