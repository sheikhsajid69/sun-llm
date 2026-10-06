"""Development launcher for LLM Prompt Project.

Canonical execution command remains:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
