# AI Agent Memory & Operational Context

This document provides persistent context, technical rationale, and architectural constraints for future AI coding agents and engineers working on the **LLM Prompt Project**.

---

## 1. Architectural Summary & Invariants
- **Framework:** FastAPI with Uvicorn ASGI server.
- **Entry Point:** Canonical startup is `uvicorn app.main:app --host 0.0.0.0 --port 8000` (or `python run.py` for local dev).
- **Configuration:** Managed via `app/config.py` using `Settings` dataclass and cached via `get_settings()`. Automatically loads `.env` if present.
- **Provider Architecture:** Factory in `app/services/llm/__init__.py` returns subclasses of `LLMProvider` (`OllamaProvider` or `HuggingFaceProvider`).
- **Memory Discipline:** Bounded conversational history managed in `app/services/conversation.py` via `collections.deque(maxlen=settings.max_history_messages)`.
- **UI:** Server-side Jinja2 template (`templates/index.html`) using vanilla JS and CSS; renders output safely using DOM `textContent`.

---

## 2. Key Engineering Decisions & Rationale
1. **Dependency Injection in Routes:**
   Routes in `app/routes/chat.py` use `Depends(get_provider)` and `Depends(get_conversation_service)`. This allows tests to override dependencies cleanly without monkey-patching globals or touching network sockets.
2. **Provider Exception Hierarchy:**
   Custom domain exceptions (`LLMConnectionError`, `LLMTimeoutError`, `LLMAuthenticationError`, `LLMModelNotFoundError`) inherit from `LLMProviderError`. Route handlers map these directly to semantic HTTP status codes (503, 504, 502, 500) while sanitizing error messages.
3. **OpenAI-Compatible Hugging Face Interface:**
   The Hugging Face provider interacts with the modern `/v1/chat/completions` endpoint standard, supporting models like `mistralai/Mistral-7B-Instruct-v0.2`. It handles cold starts (HTTP 503 with `estimated_time`) and rate limits (HTTP 429).
4. **Offline Test Suite:**
   All 26 unit tests in `tests/` run completely offline without Ollama or network connectivity by using `httpx.MockTransport` and FastAPI `dependency_overrides`.
5. **Separation of Educational Notebooks:**
   Notebooks are kept in `notebooks/` as standalone educational artifacts and have no imports from or dependencies on `app/`.

---

## 3. Important Environment Variables
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `USE_HF` | `false` | Switch between Ollama (`false`) and Hugging Face (`true`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Network address of Ollama daemon |
| `OLLAMA_MODEL` | `mistral` | Model tag for Ollama inference |
| `HF_TOKEN` | `""` | Hugging Face user access token (required if `USE_HF=true`) |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Target model repository on Hugging Face |
| `HF_BASE_URL` | `https://api-inference.huggingface.co/models` | Base URL for Hugging Face inference API |
| `MAX_HISTORY_MESSAGES` | `10` | Maximum recent turns retained per session |
| `LLM_TIMEOUT` | `120.0` | Maximum seconds to wait for model response |
| `HOST` | `0.0.0.0` | Server bind host address |
| `PORT` | `8000` | Server bind port |

---

## 4. Important Operational Commands

```bash
# 1. Environment Setup
uv venv --python 3.11 .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt

# 2. Syntax & Compilation Verification
python -m compileall -q .

# 3. Test Suite Execution
pytest -v

# 4. Local Development Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Or:
python run.py

# 5. Container Execution
docker build -t llm-prompt-project .
docker run --rm -p 8000:8000 --env-file .env llm-prompt-project
```

---

## 5. Known Limitations & Technical Debt
1. **Volatile Session Memory:**
   Session histories reside in server RAM. Server restarts or scaling across multiple worker processes will lose session context. For production scale, introduce a Redis-backed session store.
2. **Containerized Ollama Networking:**
   When running inside Docker on Windows/macOS, `localhost:11434` points to the container itself. Users must configure `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
3. **Streaming Responses:**
   Currently, responses are generated synchronously via non-streaming POST (`stream: False`). SSE (Server-Sent Events) or WebSockets should be introduced for token streaming in a future phase.

---

## 6. Future Extension Points
- **Streaming Tokens:** Implement `StreamingResponse` using Ollama's `stream: true` and HF streaming.
- **RAG Integration:** Vector store connector (e.g., ChromaDB) feeding retrieved chunks into `ConversationService.build_messages_for_llm`.
- **Multi-Model Provider:** Expand `app/services/llm/` to support vLLM, llama.cpp, or Open-WebUI.
