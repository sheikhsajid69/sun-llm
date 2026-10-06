# LLM Prompt Project

A production-grade, modular, and privacy-first conversational AI web application built with **Python 3.11+**, **FastAPI**, and **open-weight Large Language Models** (Ollama and Hugging Face Inference).

---

## Overview

The **LLM Prompt Project** provides an end-to-end reference implementation for serving multi-turn LLM conversations without reliance on paid, proprietary APIs (such as OpenAI, Anthropic, or Gemini). It supports local offline inference via **Ollama** by default, with an optional cloud inference adapter for **Hugging Face Inference API**, all accessible through a responsive, dark-mode browser interface and a clean REST API.

---

## Problem Statement

Deploying Generative AI applications frequently incurs substantial recurring API costs, vendor lock-in, and data privacy concerns associated with commercial cloud endpoints. Furthermore, beginner and prototype LLM projects often suffer from architectural shortcomings: mixing HTTP logic directly with provider SDK calls, allowing unbounded conversational history to consume excessive memory, lacking input validation, and failing to provide automated test suites.

---

## Objectives

- **Zero-Cost Operation:** Run completely free using local open-weight models (Mistral, LLaMA 3, Phi).
- **Clean Architecture:** Strict separation of presentation, routing, domain logic, and provider integration.
- **Provider Interchangeability:** Switch between local Ollama and Hugging Face Inference via configuration without altering application code.
- **Defensive Engineering:** Strict Pydantic v2 payload validation, sanitized error codes (no leaked tracebacks), bounded ring-buffer memory, and XSS-safe frontend rendering.
- **Zero-Dependency Testing:** 100% offline unit and integration test suite using mock transports.
- **Container Readiness:** Multi-layer, unprivileged Docker containerization.

---

## Features

- **Multi-Turn Context Memory:** Server-side bounded conversational memory (`MAX_HISTORY_MESSAGES=10`) using thread-safe sliding buffers.
- **Prompt Templating:** Centralized system prompt management (`app/prompts/system.txt`) with automatic history assembly.
- **Dynamic Sampling Control:** Interactive temperature slider (0.0 to 2.0) supported in both the Web UI and the API.
- **Modern Responsive Web UI:** Dark-mode interface with message bubbles, typing indicators, live health polling, and keyboard shortcuts (`Enter` to send, `Shift+Enter` for newline).
- **Health & Diagnostic Probing:** `/health` endpoint distinguishes between overall application health and provider reachability.
- **Dual Inference Providers:** Native asynchronous adapters for local Ollama and cloud-based Hugging Face chat completions.

---

## Architecture

The system follows a layered, hexagonal-inspired architecture:

```mermaid
flowchart TD
    subgraph Presentation ["Presentation Layer"]
        UI["Web UI (templates/index.html)"]
        FastAPI["FastAPI App (app/main.py)"]
    end

    subgraph Routing ["Routing & Control Layer"]
        Router["Chat Router (app/routes/chat.py)"]
        Validation["Pydantic Models (app/models.py)"]
    end

    subgraph Domain ["Domain & Services Layer"]
        ConvService["ConversationService (app/services/conversation.py)"]
        SysPrompt["System Prompt (app/prompts/system.txt)"]
    end

    subgraph Infrastructure ["Provider Infrastructure Layer"]
        Factory["Provider Factory (app/services/llm/__init__.py)"]
        BaseProv["LLMProvider ABC (app/services/llm/base.py)"]
        OllamaProv["OllamaProvider (app/services/llm/ollama.py)"]
        HFProv["HuggingFaceProvider (app/services/llm/huggingface.py)"]
    end

    subgraph External ["External LLM Engines"]
        OllamaEngine["Local Ollama Daemon (:11434)"]
        HFEngine["Hugging Face API Cloud"]
    end

    UI -->|"HTTP POST /api/chat"| Router
    UI -->|"HTTP GET /health"| Router
    Router --> Validation
    Router --> ConvService
    ConvService --> SysPrompt
    Router --> Factory
    Factory --> BaseProv
    BaseProv <|-- OllamaProv
    BaseProv <|-- HFProv
    OllamaProv -->|"Async HTTP"| OllamaEngine
    HFProv -->|"Async HTTP"| HFEngine
```

---

## Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Strong typing, high-performance async runtime, cross-platform |
| **Framework** | FastAPI 0.109+ | High-throughput asynchronous REST API framework |
| **ASGI Server** | Uvicorn (standard) | Production-ready ASGI server with uvloop support |
| **Data Validation** | Pydantic v2 | Robust schema validation and sanitization |
| **HTTP Client** | HTTPX (async) | Non-blocking HTTP client with connection pooling and timeouts |
| **Templating** | Jinja2 | Fast, safe server-side HTML template rendering |
| **Frontend** | HTML5 / CSS3 / Vanilla JS | Zero-dependency, lightweight, native DOM manipulation |
| **Testing** | Pytest & Pytest-Asyncio | Automated testing framework with async support |
| **Packaging** | Docker | Minimal multi-layer image based on `python:3.11-slim` |

---

## Project Structure

```text
llm-prompt-project/
│
├── app/
│   ├── __init__.py                # Package root & version
│   ├── main.py                    # FastAPI application factory & lifespan
│   ├── config.py                  # Centralized settings container
│   ├── models.py                  # Pydantic schemas (Chat, Health, Errors)
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py                # REST endpoints: /api/chat, /health
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── conversation.py        # Bounded conversation memory & prompt assembly
│   │   │
│   │   └── llm/
│   │       ├── __init__.py        # Provider factory (get_llm_provider)
│   │       ├── base.py            # Abstract LLMProvider interface & custom exceptions
│   │       ├── ollama.py          # Local Ollama inference provider
│   │       └── huggingface.py     # Hugging Face inference provider
│   │
│   └── prompts/
│       └── system.txt             # Immutable base system instructions
│
├── templates/
│   └── index.html                 # Accessible dark-mode chat interface
│
├── tests/
│   ├── __init__.py
│   ├── test_health.py             # UI serving and health probe tests
│   ├── test_chat.py               # API request validation and error mapping tests
│   ├── test_conversation.py       # Bounded memory, roles, and session isolation tests
│   └── test_providers.py          # Direct provider mock transport unit tests
│
├── notebooks/                     # Educational NLP artifacts (decoupled from backend)
│   ├── module_6_part_1.ipynb      # NLTK tokenization & lemmatization
│   ├── module_6_part_2.ipynb      # BoW & TF-IDF vectorization
│   └── module_6_part_3.ipynb      # Sentiment classification with Gradio
│
├── docs/
│   ├── PRD.md                     # Product Requirements Document
│   ├── Architecture.md            # Architectural specifications & diagrams
│   ├── Rules.md                   # Engineering standards for developers & AI agents
│   ├── Phases.md                  # Detailed implementation roadmap
│   └── Memory.md                  # Persistent AI agent operational context
│
├── .env.example                   # Sanitized configuration template
├── .gitignore                     # Git ignore rules
├── Dockerfile                     # Multi-layer secure container image
├── .dockerignore                  # Docker build exclusions
├── requirements.txt               # Pinned runtime and testing dependencies
├── pytest.ini                     # Pytest runner configuration
├── README.md                      # Complete project documentation
└── run.py                         # Convenience development entrypoint
```

---

## Prerequisites

- **Python:** 3.11 or higher
- **Ollama (Optional, for local inference):** Downloaded from [ollama.com](https://ollama.com)
- **Hugging Face Account (Optional, for cloud inference):** Free account at [huggingface.co](https://huggingface.co)
- **Docker (Optional, for containerized execution):** Docker Engine 20.10+

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd llm-prompt-project
```

### 2. Create and Activate Virtual Environment
```bash
# Using standard Python:
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

*(Alternatively, using `uv`: `uv venv --python 3.11 .venv && .venv\Scripts\activate`)*

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Variables

Copy the provided template to configure the application:

```bash
cp .env.example .env
```

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `USE_HF` | `bool` | `false` | Set to `true` to enable Hugging Face cloud inference |
| `OLLAMA_BASE_URL` | `string` | `http://localhost:11434` | Network address of local Ollama daemon |
| `OLLAMA_MODEL` | `string` | `mistral` | Model tag to load in Ollama |
| `HF_TOKEN` | `string` | `""` | User Access Token from Hugging Face (required if `USE_HF=true`) |
| `HF_MODEL` | `string` | `mistralai/Mistral-7B-Instruct-v0.2` | Target model repository on Hugging Face |
| `HF_BASE_URL` | `string` | `https://api-inference.huggingface.co/models` | Base URL for Hugging Face inference API |
| `MAX_HISTORY_MESSAGES` | `int` | `10` | Upper bound of retained multi-turn messages |
| `LLM_TIMEOUT` | `float` | `120.0` | Maximum network timeout in seconds for generation |
| `HOST` | `string` | `0.0.0.0` | Bind host address |
| `PORT` | `int` | `8000` | Bind port number |

---

## Ollama Setup (Local Inference)

1. **Install Ollama:** Visit [ollama.com](https://ollama.com) and install the native application.
2. **Start the Ollama daemon:**
   ```bash
   ollama serve
   ```
3. **Pull your target model:**
   ```bash
   ollama pull mistral
   # Or lightweight alternatives:
   ollama pull llama3
   ollama pull phi3
   ```
4. Configure `.env` with `USE_HF=false` and `OLLAMA_MODEL=mistral`.

---

## Hugging Face Setup (Cloud Inference)

1. Create a free account at [Hugging Face](https://huggingface.co).
2. Generate an Access Token under **Settings > Access Tokens** with `read` permission.
3. Configure your `.env`:
   ```env
   USE_HF=true
   HF_TOKEN=hf_YourSecretTokenHere
   HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   ```

---

## Running Locally

### Canonical Startup (Uvicorn)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Development Launcher
```bash
python run.py
```

Once running, navigate to **http://localhost:8000** to interact with the web chat interface.

---

## API Documentation

Interactive OpenAPI documentation is generated automatically by FastAPI:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Key Endpoints

#### 1. Web User Interface
- **Endpoint:** `GET /`
- **Response:** `text/html` (The interactive chat application).

#### 2. Health & Diagnostic Check
- **Endpoint:** `GET /health`
- **Response (HTTP 200):**
  ```json
  {
    "status": "ok",
    "provider": "ollama",
    "model": "mistral",
    "details": {
      "reachable": true,
      "model_available": true,
      "available_models": ["mistral:latest"]
    }
  }
  ```
  *(If Ollama is offline, returns status `"degraded"` with diagnostic error details while keeping HTTP 200).*

#### 3. Chat Interaction
- **Endpoint:** `POST /api/chat`
- **Request Body:**
  ```json
  {
    "message": "Explain recursion in three sentences.",
    "temperature": 0.7,
    "session_id": "default",
    "history": []
  }
  ```
- **Response (HTTP 200):**
  ```json
  {
    "reply": "Recursion is a programming technique where a function calls itself...",
    "model_used": "mistral",
    "provider": "ollama",
    "history": [
      {"role": "user", "content": "Explain recursion in three sentences."},
      {"role": "assistant", "content": "Recursion is a programming technique..."}
    ]
  }
  ```
- **Error Responses:**
  - `422 Unprocessable Entity`: Validation failure (empty prompt, invalid temperature, or unsupported role).
  - `502 Bad Gateway`: Model not found or provider authorization rejected.
  - `503 Service Unavailable`: Local Ollama daemon unreachable or model loading.
  - `504 Gateway Timeout`: Inference exceeded `LLM_TIMEOUT`.

---

## Testing

The test suite runs **100% offline** without needing live LLM daemons, API tokens, or network connectivity.

Run all tests via pytest:

```bash
pytest -v
```

Verify compilation across the entire project:

```bash
python -m compileall -q .
```

### Test Coverage Highlights
- **`test_health.py`:** Verifies HTML UI delivery and both healthy and degraded diagnostic states.
- **`test_chat.py`:** Verifies successful chat flow, input validation (empty message, whitespace, temperature boundaries, role enumeration), and error mapping (504, 503, 502).
- **`test_conversation.py`:** Verifies bounded memory capping (`deque(maxlen=N)`), system prompt preservation, client history filtering, and session isolation.
- **`test_providers.py`:** Verifies provider adapters directly using HTTPX `MockTransport`.

---

## Docker

### Build the Image
```bash
docker build -t llm-prompt-project .
```

### Run Container with Local Ollama
> **Note on Ollama Networking:** When running Docker on Windows or macOS, `localhost` inside the container refers to the container itself. To connect to an Ollama daemon running on the host machine, set `OLLAMA_BASE_URL` to `http://host.docker.internal:11434`.

```bash
docker run --rm -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=mistral \
  llm-prompt-project
```

### Run Container with Hugging Face
```bash
docker run --rm -p 8000:8000 \
  -e USE_HF=true \
  -e HF_TOKEN=your_token_here \
  -e HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
  llm-prompt-project
```

---

## Deployment

### Production Checklist
1. Deploy behind a reverse proxy (Nginx, Traefik, or Caddy) handling TLS termination.
2. In multi-replica deployments (Kubernetes, AWS ECS), replace the in-memory conversation store with a shared Redis store.
3. Configure `LLM_TIMEOUT` to match expected model inference speeds and server worker timeouts.
4. Set explicit resource limits (`memory: 512Mi`, `cpu: 1.0`) in your container orchestrator.

---

## Security

- **Zero Secrets Committed:** Credentials are read strictly from runtime environment variables.
- **XSS Mitigation:** Frontend DOM updates utilize `textContent` assignment, preventing script execution from model output.
- **Payload Clamping:** Prompts are restricted to 4,000 characters and history items to 10,000 characters.
- **Sanitized Error Responses:** Internal stack traces, database strings, and provider credentials are never surfaced to clients.
- **Unprivileged Container:** Dockerfile executes as `appuser` (UID 1000).

---

## Limitations

- **In-Memory Memory Persistence:** Conversation history is stored in RAM and resets upon server restart.
- **Non-Streaming Generation:** Responses are delivered in full after completion; token-level streaming (SSE) is slated for future releases.
- **Hugging Face Rate Limits:** Free-tier Hugging Face models are subject to concurrent request throttling and cold starts.

---

## Future Roadmap

- [ ] Server-Sent Events (SSE) streaming for real-time token rendering.
- [ ] Persistent conversation memory backend with Redis or SQLite.
- [ ] Retrieval-Augmented Generation (RAG) pipeline integration with local vector store.
- [ ] Multi-turn session management with exportable chat histories (JSON/Markdown).
- [ ] Support for llama.cpp and vLLM inference backends.

---

## Internship Project Context

This project was engineered as part of an **Advanced Generative AI & Software Architecture Internship**. It demonstrates:
- Professional refactoring of legacy prototype code into clean production architecture.
- Full mastery of Python async paradigms and FastAPI dependency injection.
- Defensive API design and strict data contract enforcement.
- Practical local LLM integration and open-weight model deployment.
- Automated testing methodologies isolating external dependencies.

---

## License

This project is open-source and distributed under the [MIT License](LICENSE).
