"""API routers module."""

from .ocr import router as ocr_router
from .health import router as health_router
from .tasks import router as tasks_router

__all__ = [
    "ocr_router",
    "health_router",
    "tasks_router",
]
