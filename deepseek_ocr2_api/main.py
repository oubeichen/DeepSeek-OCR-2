"""
DeepSeek-OCR-2 API Server - Main Application

FastAPI-based API server for DeepSeek-OCR-2 model inference.
Supports PDF and image OCR with configurable parameters.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from .config import get_settings
from .engine.manager import EngineManager
from .routers import ocr_router, health_router, tasks_router
from .task_manager import TaskManager
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting DeepSeek-OCR-2 API Server...")
    settings = get_settings()

    # Apply environment variables
    settings.apply_env_vars()

    # Initialize engine
    try:
        manager = EngineManager.get_instance()
        logger.info("Initializing async engine...")
        manager.initialize(settings)
        logger.info("Engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        logger.warning("Server will start but OCR endpoints will not work until engine is initialized.")

    # Start task worker
    task_manager = TaskManager.get_instance()
    task_manager.start_worker()
    logger.info("Task worker started.")

    yield

    # Shutdown
    logger.info("Shutting down DeepSeek-OCR-2 API Server...")

    # Stop task worker
    task_manager = TaskManager.get_instance()
    task_manager.stop_worker()

    try:
        manager = EngineManager.get_instance()
        manager.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("Server shutdown complete.")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="DeepSeek-OCR-2 API",
        description="""
## DeepSeek-OCR-2 API Server

A high-performance API server for document OCR using the DeepSeek-OCR-2 model.

### Features

- **Single Image OCR**: Process individual images and extract text as markdown
- **PDF OCR**: Process PDF documents with multi-page support
- **Batch Processing**: Process multiple images in a single request
- **Configurable Parameters**: Customize prompts, sampling parameters, and processing options
- **Result Packaging**: Download results as ZIP files with markdown, annotated images, and extracted content

### Supported Formats

- **Images**: PNG, JPG, JPEG, WebP, BMP, TIFF, GIF
- **Documents**: PDF

### Quick Start

1. Upload a file to `/api/v1/ocr/image` or `/api/v1/ocr/pdf`
2. Optionally customize parameters via form fields
3. Download the resulting ZIP file with extracted content

### Configuration

All parameters can be configured via:
- Environment variables (prefix: `DEEPSEEK_OCR2_`)
- Command-line arguments
- API request parameters (override defaults per-request)
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(ocr_router)
    app.include_router(tasks_router)

    # Get static files directory
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "InternalServerError",
                "message": str(exc),
            }
        )

    # Root endpoint - serve web interface
    @app.get("/", tags=["Root"])
    async def root():
        """
        Serve the web interface.
        """
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file, media_type="text/html")
        return {
            "name": "DeepSeek-OCR-2 API",
            "version": __version__,
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
        }

    @app.get("/favicon.ico")
    async def favicon():
        icon_file = static_dir / "favicon.ico"
        if icon_file.exists():
            return FileResponse(icon_file, media_type="image/x-icon")
        raise HTTPException(status_code=404, detail="Favicon not found")

    return app


# Create the application instance
app = create_app()


def custom_openapi():
    """
    Customize OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="DeepSeek-OCR-2 API",
        version=__version__,
        description=app.description,
        routes=app.routes,
    )

    # Add server information
    openapi_schema["servers"] = [
        {"url": "/", "description": "Current server"},
    ]

    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "OCR",
            "description": "OCR processing endpoints for images and PDFs",
        },
        {
            "name": "Health & Config",
            "description": "Health check and configuration endpoints",
        },
        {
            "name": "Root",
            "description": "Root endpoint",
        },
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
