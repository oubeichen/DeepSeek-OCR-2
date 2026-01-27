"""
Health check and configuration routes for DeepSeek-OCR-2 API Server.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..engine.manager import EngineManager
from ..schemas.response import HealthResponse, ConfigResponse, EngineStatusResponse
from .. import __version__

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Health & Config"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API server and inference engine.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the server and whether the inference engine
    is initialized and ready to process requests.
    """
    manager = EngineManager.get_instance()

    status = "healthy" if manager.is_initialized() else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.utcnow(),
        engine_initialized=manager.is_initialized(),
        engine_mode=manager.get_mode(),
    )


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get Configuration",
    description="Get the current server configuration settings.",
)
async def get_config() -> ConfigResponse:
    """
    Get current configuration.

    Returns all configurable settings currently in use by the server.
    """
    settings = get_settings()

    return ConfigResponse(
        model_path=settings.model_path,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        tensor_parallel_size=settings.tensor_parallel_size,
        max_model_len=settings.max_model_len,
        engine_mode=settings.engine_mode,
        image_size=settings.image_size,
        base_size=settings.base_size,
        min_crops=settings.min_crops,
        max_crops=settings.max_crops,
        crop_mode=settings.crop_mode,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        ngram_size=settings.ngram_size,
        window_size=settings.window_size,
        pdf_dpi=settings.pdf_dpi,
        host=settings.host,
        port=settings.port,
    )


@router.get(
    "/engine/status",
    response_model=EngineStatusResponse,
    summary="Engine Status",
    description="Get the current status of the inference engine.",
)
async def get_engine_status() -> EngineStatusResponse:
    """
    Get engine status.

    Returns detailed information about the inference engine state.
    """
    manager = EngineManager.get_instance()
    settings = manager.get_settings()

    return EngineStatusResponse(
        initialized=manager.is_initialized(),
        mode=manager.get_mode(),
        model_path=settings.model_path if settings else None,
        gpu_memory_utilization=settings.gpu_memory_utilization if settings else None,
    )


@router.post(
    "/engine/initialize",
    response_model=EngineStatusResponse,
    summary="Initialize Engine",
    description="Manually initialize the inference engine. Usually not needed as engine initializes on startup.",
)
async def initialize_engine() -> EngineStatusResponse:
    """
    Initialize the inference engine.

    This endpoint is typically not needed as the engine initializes automatically
    on server startup. Use this only if you need to reinitialize after a shutdown.
    """
    manager = EngineManager.get_instance()

    if manager.is_initialized():
        raise HTTPException(
            status_code=400,
            detail="Engine is already initialized. Restart the server to reinitialize."
        )

    settings = get_settings()

    try:
        manager.initialize(settings, mode=settings.engine_mode)
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize engine: {str(e)}"
        )

    return EngineStatusResponse(
        initialized=manager.is_initialized(),
        mode=manager.get_mode(),
        model_path=settings.model_path,
        gpu_memory_utilization=settings.gpu_memory_utilization,
    )
