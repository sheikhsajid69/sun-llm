# Engineering Rules & Guidelines

These engineering rules govern all ongoing development, refactoring, and code contributions to the **LLM Prompt Project**. They serve as strict operational constraints for human developers and future AI coding agents.

## 1. Zero Secrets Policy
- **Rule 1.1:** Never hardcode secrets, access tokens, API keys, or private endpoint URLs into Python files, HTML templates, JavaScript files, Dockerfiles, or documentation.
- **Rule 1.2:** Secrets must only be ingested via environment variables (e.g., `HF_TOKEN`).
- **Rule 1.3:** Never commit real `.env` files to source control. Only commit `.env.example` with sanitized placeholder values.
- **Rule 1.4:** Never log sensitive credentials, authorization headers, or private tokens in structured logging statements.

## 2. Clean Repository Hygiene
- **Rule 2.1:** Never commit virtual environments (`venv/`, `.venv/`), bytecode caches (`__pycache__/`, `*.pyc`), OS metadata (`.DS_Store`, `__MACOSX/`), or test artifacts (`.pytest_cache/`, `.coverage`).
- **Rule 2.2:** Maintain comprehensive `.gitignore` and `.dockerignore` patterns reflecting these exclusions.
- **Rule 2.3:** Keep educational artifacts (e.g., Jupyter notebooks) isolated in `notebooks/` and decoupled from production backend imports.

## 3. Strict Provider Isolation
- **Rule 3.1:** Route controllers (`app/routes/`) must never construct provider-specific payloads, handle provider HTTP status codes, or issue provider network requests directly.
- **Rule 3.2:** All provider interactions must be encapsulated inside classes implementing `LLMProvider` (`app/services/llm/base.py`).
- **Rule 3.3:** Adding a new LLM provider must only require implementing a new adapter subclass without modifying route handlers.

## 4. Rigorous Input Validation
- **Rule 4.1:** All client inputs must be validated via Pydantic models before reaching domain logic.
- **Rule 4.2:** Reject empty or whitespace-only prompts with HTTP 422.
- **Rule 4.3:** Enforce strict boundaries on sampling temperatures (`0.0 <= temperature <= 2.0`).
- **Rule 4.4:** Validate message roles strictly against allowed enumeration values (`user`, `assistant`, `system`).
- **Rule 4.5:** Enforce payload length limits (`max_length=4000` for user prompts; `max_length=10000` for history messages) to guard against denial-of-service and memory exhaustion.

## 5. Bounded Memory Discipline
- **Rule 5.1:** Conversation history must never grow unbounded. Always enforce a hard ceiling (`MAX_HISTORY_MESSAGES`, default 10) using bounded ring buffers (`collections.deque(maxlen=N)`).
- **Rule 5.2:** Client-provided history must be sanitized and truncated server-side before ingestion.
- **Rule 5.3:** Session memory stores must be thread-safe.

## 6. Asynchronous Correctness
- **Rule 6.1:** All network I/O operations (Ollama and Hugging Face HTTP calls) must use non-blocking asynchronous HTTP clients (`httpx.AsyncClient`).
- **Rule 6.2:** All external network calls must enforce explicit timeouts (`LLM_TIMEOUT`). Never issue un-bounded HTTP requests.
- **Rule 6.3:** Use async context managers for resource lifecycle management (`lifespan` in FastAPI).

## 7. Robust Error Handling & Fault Isolation
- **Rule 7.1:** External provider failures (unreachable daemon, model missing, timeout) must not crash the application server.
- **Rule 7.2:** Translate provider network failures into semantic HTTP responses:
  - Timeouts -> HTTP 504 Gateway Timeout
  - Unreachable service -> HTTP 503 Service Unavailable
  - Auth failure / missing model -> HTTP 502 Bad Gateway
  - Missing server credentials -> HTTP 500 Internal Server Error
- **Rule 7.3:** Never expose internal Python tracebacks or filesystem paths to client responses.

## 8. Test Suite Independence
- **Rule 8.1:** Unit tests must pass 100% offline without live Ollama instances, Hugging Face tokens, or internet access.
- **Rule 8.2:** Mock external dependencies using FastAPI dependency overrides or HTTPX mock transports.
- **Rule 8.3:** Every PR or code modification must pass `pytest` and `python -m compileall .`.

## 9. Dependency & Architecture Simplicity
- **Rule 9.1:** Avoid over-engineering. Do not introduce microservices, distributed message brokers (Kafka/Celery), or complex ORMs unless concrete production requirements mandate them.
- **Rule 9.2:** Pin dependency version boundaries in `requirements.txt` to prevent breaking changes while enabling security patches.
