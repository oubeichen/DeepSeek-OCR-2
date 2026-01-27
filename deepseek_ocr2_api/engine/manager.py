"""
Engine Manager for DeepSeek-OCR-2 API Server.

Implements singleton pattern to ensure the model is loaded only once
and reused across all requests.
"""

import os
import sys
import threading
import logging
from typing import Optional, Union, Literal

# Add the DeepSeek-OCR2-vllm directory to path for imports
VLLM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "DeepSeek-OCR2-master", "DeepSeek-OCR2-vllm")
if VLLM_DIR not in sys.path:
    sys.path.insert(0, VLLM_DIR)

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class EngineManager:
    """
    Singleton Engine Manager for vLLM engine.

    Ensures the model is loaded only once and provides thread-safe access
    to the engine instance.

    Usage:
        manager = EngineManager.get_instance()
        manager.initialize(settings, mode="sync")
        engine = manager.get_engine()
    """

    _instance: Optional["EngineManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._engine = None
                    cls._instance._mode = None
                    cls._instance._settings = None
                    cls._instance._processor = None
        return cls._instance

    @classmethod
    def get_instance(cls) -> "EngineManager":
        """Get the singleton instance of EngineManager."""
        return cls()

    def is_initialized(self) -> bool:
        """Check if the engine has been initialized."""
        return self._initialized and self._engine is not None

    def get_mode(self) -> Optional[str]:
        """Get the current engine mode."""
        return self._mode

    def get_settings(self) -> Optional[Settings]:
        """Get the current settings."""
        return self._settings

    def initialize(
        self,
        settings: Optional[Settings] = None,
        mode: Literal["sync", "async"] = "sync"
    ) -> None:
        """
        Initialize the vLLM engine.

        Args:
            settings: Configuration settings. If None, uses global settings.
            mode: Engine mode - "sync" for LLM, "async" for AsyncLLMEngine.

        Raises:
            RuntimeError: If engine is already initialized with different settings.
        """
        with self._lock:
            if self._initialized:
                logger.warning("Engine already initialized. Skipping re-initialization.")
                return

            settings = settings or get_settings()
            self._settings = settings
            self._mode = mode

            # Apply environment variables
            settings.apply_env_vars()

            logger.info(f"Initializing engine in {mode} mode...")
            logger.info(f"Model path: {settings.model_path}")
            logger.info(f"GPU memory utilization: {settings.gpu_memory_utilization}")
            logger.info(f"Tensor parallel size: {settings.tensor_parallel_size}")

            # Register the custom model
            self._register_model()

            # Initialize the engine based on mode
            if mode == "sync":
                self._init_sync_engine(settings)
            else:
                self._init_async_engine(settings)

            # Initialize the processor
            self._init_processor(settings)

            self._initialized = True
            logger.info("Engine initialization complete.")

    def _register_model(self) -> None:
        """Register the DeepseekOCR2ForCausalLM model with vLLM."""
        from vllm.model_executor.models.registry import ModelRegistry
        from deepseek_ocr2 import DeepseekOCR2ForCausalLM

        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
        logger.info("Registered DeepseekOCR2ForCausalLM model.")

    def _init_sync_engine(self, settings: Settings) -> None:
        """Initialize synchronous LLM engine."""
        from vllm import LLM

        self._engine = LLM(
            model=settings.model_path,
            hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
            dtype=settings.dtype,
            block_size=settings.block_size,
            enforce_eager=settings.enforce_eager,
            trust_remote_code=settings.trust_remote_code,
            max_model_len=settings.max_model_len,
            swap_space=settings.swap_space,
            max_num_seqs=settings.max_num_seqs,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            disable_mm_preprocessor_cache=settings.disable_mm_preprocessor_cache,
        )
        logger.info("Synchronous LLM engine initialized.")

    def _init_async_engine(self, settings: Settings) -> None:
        """Initialize asynchronous LLM engine."""
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=settings.model_path,
            hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
            dtype=settings.dtype,
            max_model_len=settings.max_model_len,
            enforce_eager=settings.enforce_eager,
            trust_remote_code=settings.trust_remote_code,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Asynchronous LLM engine initialized.")

    def _init_processor(self, settings: Settings) -> None:
        """Initialize the image processor."""
        from process.image_process import DeepseekOCR2Processor

        self._processor = DeepseekOCR2Processor()
        logger.info("Image processor initialized.")

    def get_engine(self) -> Union["LLM", "AsyncLLMEngine"]:
        """
        Get the vLLM engine instance.

        Returns:
            The initialized engine (LLM or AsyncLLMEngine).

        Raises:
            RuntimeError: If engine is not initialized.
        """
        if not self._initialized or self._engine is None:
            raise RuntimeError(
                "Engine not initialized. Call initialize() first."
            )
        return self._engine

    def get_processor(self) -> "DeepseekOCR2Processor":
        """
        Get the image processor instance.

        Returns:
            The initialized processor.

        Raises:
            RuntimeError: If processor is not initialized.
        """
        if not self._initialized or self._processor is None:
            raise RuntimeError(
                "Processor not initialized. Call initialize() first."
            )
        return self._processor

    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        with self._lock:
            if self._engine is not None:
                logger.info("Shutting down engine...")
                # vLLM engines don't have explicit shutdown, but we clear references
                self._engine = None
                self._processor = None
                self._initialized = False
                self._mode = None
                self._settings = None
                logger.info("Engine shutdown complete.")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None
