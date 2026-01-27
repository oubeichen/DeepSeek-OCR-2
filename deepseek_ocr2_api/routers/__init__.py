"""API routers module."""

from .ocr import router as ocr_router
from .health import router as health_router

__all__ = [
    "ocr_router",
    "health_router",
]
