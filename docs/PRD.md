# Product Requirements Document (PRD)

## 1. Product Vision
The **LLM Prompt Project** is a modular, lightweight, free, and privacy-conscious AI conversational platform. It delivers multi-turn conversational AI capabilities to local developer workstations and self-hosted environments using open-source models—completely free of paid third-party APIs (OpenAI, Anthropic, Gemini).

## 2. Problem Statement
Many generative AI applications rely on paid proprietary cloud APIs that pose privacy risks, incur recurring costs, and fail in air-gapped or offline development environments. Students, interns, and developers need an accessible, well-engineered, and reproducible reference architecture for LLM prompt engineering, multi-turn context retention, and backend inference orchestration using free, open-weight language models.

## 3. Goals & Objectives
- **Zero Cost Inference:** Default to local Ollama inference without subscription costs or API keys.
- **Provider Interchangeability:** Support seamless switching between local Ollama and cloud-based Hugging Face Inference without modifying application logic.
- **Architectural Rigor:** Demonstrate production-grade software engineering patterns—separation of concerns, dependency injection, strict data validation, and automated testing.
- **Bounded Context Memory:** Maintain coherent multi-turn conversations while preventing memory leaks and unbounded payload inflation.
- **Containerization & Deployment Readiness:** Provide container definitions and configuration isolation for standard DevOps deployment.

## 4. Target Users
- **Internship Evaluators & Reviewers:** Assessing technical competence in Python, FastAPI, API architecture, and DevOps.
- **Developers & Students:** Learning prompt engineering, LLM integration, and FastAPI backend patterns.
- **Privacy-Conscious Users:** Running local conversational models on private data.

## 5. Functional Requirements

### FR-1: Conversational Chat API (`POST /api/chat`)
- Accepts a user prompt (`message`), optional sampling `temperature` (0.0 to 2.0), optional client `history`, and `session_id`.
- Validates prompt content (non-empty, length bounded to 4,000 characters).
- Formats messages with an immutable system prompt, bounded historical turns, and current user input.
- Invokes the active LLM provider and returns the assistant's reply, the model utilized, the provider name, and updated conversation history.

### FR-2: Health & Diagnostics (`GET /health`)
- Evaluates application operational status and probes the configured LLM provider reachability.
- Distinguishes application health (`ok`) from provider offline status (`degraded`).
- Returns HTTP 200 with actionable diagnostic information without crashing.

### FR-3: Interactive Web User Interface (`GET /`)
- Serves a lightweight, responsive web chat interface.
- Provides real-time chat bubbles, role differentiation, typing indicators, and backend status feedback.
- Includes interactive controls for sampling temperature adjustment and session memory clearing.
- Renders assistant output safely via standard DOM text nodes, mitigating cross-site scripting (XSS).

### FR-4: Context Memory & Session Handling
- Maintains an in-memory conversational history per session.
- Enforces strict upper bounds (`MAX_HISTORY_MESSAGES`, default: 10) using thread-safe sliding buffers (`collections.deque`).
- Filters and sanitizes client-supplied history, permitting only allowed roles (`user`, `assistant`).

### FR-5: Dual Provider Support
- **Ollama Provider:** Communicates with local daemon over async HTTP (`/api/chat` and `/api/tags`).
- **Hugging Face Provider:** Communicates with Hugging Face Inference API via OpenAI-compatible chat endpoints (`v1/chat/completions`).

## 6. Non-Functional Requirements
- **Reliability & Error Isolation:** External provider failures (timeouts, connection drops, model absences) must translate into standard HTTP status codes (502, 503, 504) with sanitised error messages. Never expose stack traces or raw tokens.
- **Performance:** Sub-millisecond routing overhead; network calls bounded by configurable timeouts (`LLM_TIMEOUT`, default: 120s).
- **Security:** Zero hardcoded credentials; credentials sourced exclusively from environment variables; inputs bounded and sanitized.
- **Testability:** Unit test suite must execute completely offline with zero dependencies on live LLM daemons or external network connections.
- **Cross-Platform Compatibility:** Clean execution on Linux, macOS, and Windows.

## 7. Constraints & Assumptions
- In-memory conversation state does not persist across server restarts.
- GPU acceleration depends on the host machine running the Ollama daemon.
- Hugging Face free-tier inference is subject to provider rate limits (HTTP 429) and cold-start model loading delays (HTTP 503).

## 8. Non-Goals
- Multi-user authentication, OAuth2, or user account databases (out of scope for v1).
- Persistent distributed caching (e.g., Redis) or relational database persistence.
- Retrieval-Augmented Generation (RAG) vector pipelines (planned for future phases).
- Proprietary paid LLM provider integrations (OpenAI, Anthropic).
