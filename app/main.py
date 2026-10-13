"""FastAPI Application Entry Point.

Configures application lifespan, routes, template rendering, logging,
and global exception handling.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.routes.chat import router as chat_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_prompt_project")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle context manager."""
    settings = get_settings()
    logger.info("Initializing LLM Prompt Project...")
    logger.info("Configured Provider: %s", "huggingface" if settings.use_hf else "ollama")
    logger.info(
        "Configured Model: %s",
        settings.hf_model if settings.use_hf else settings.ollama_model,
    )
    logger.info("Memory History Limit: %d messages", settings.max_history_messages)
    logger.info("Templates Directory: %s", settings.templates_dir)
    yield
    logger.info("Shutting down LLM Prompt Project...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for FastAPI instance."""
    app_settings = settings or get_settings()

    application = FastAPI(
        title="LLM Prompt Project",
        description="Production-ready FastAPI chatbot with Ollama and Hugging Face providers",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Initialize Jinja2 templates using resolved directory path
    templates = Jinja2Templates(directory=str(app_settings.templates_dir))

    # Mount Chat & Health API Routes
    application.include_router(chat_router)

    FAVICON_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<defs>'
        '<linearGradient id="sunGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#5266eb"/>'
        '<stop offset="50%" stop-color="#7a8aff"/>'
        '<stop offset="100%" stop-color="#9cb4e8"/>'
        '</linearGradient>'
        '<linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#5266eb" stop-opacity="0.8"/>'
        '<stop offset="100%" stop-color="#edecfb" stop-opacity="0.2"/>'
        '</linearGradient>'
        '</defs>'
        '<rect width="32" height="32" rx="8" fill="#12121b"/>'
        '<circle cx="16" cy="16" r="11" fill="none" stroke="url(#glowGrad)" stroke-width="1.5" stroke-dasharray="3 2"/>'
        '<circle cx="16" cy="16" r="7.5" fill="url(#sunGrad)"/>'
        '<circle cx="16" cy="16" r="3" fill="#ffffff" fill-opacity="0.9"/>'
        '</svg>'
    )

    @application.get("/favicon.ico", include_in_schema=False)
    async def get_favicon() -> Response:
        """Serve application favicon as SVG."""
        return Response(content=FAVICON_SVG, media_type="image/svg+xml")

    # Root Web UI endpoint
    @application.get("/", response_class=HTMLResponse, tags=["ui"])
    async def render_chat_ui(request: Request) -> HTMLResponse:
        """Render the web browser chat user interface."""
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": "SUN-LLM",
                "provider": "huggingface" if app_settings.use_hf else "ollama",
                "model": (
                    app_settings.hf_model
                    if app_settings.use_hf
                    else app_settings.ollama_model
                ),
            },
        )

    # Global Exception Handlers to sanitize unexpected errors
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )

    return application


# Global ASGI application instance for Uvicorn
app = create_app()
