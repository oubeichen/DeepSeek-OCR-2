"""
Configuration management for DeepSeek-OCR-2 API Server.

Supports environment variables, .env files, and command-line arguments.
All configuration options can be overridden at runtime via API parameters.
"""

import os
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be configured via:
    - Environment variables (prefix: DEEPSEEK_OCR2_)
    - .env file
    - Command-line arguments (via argparse in __main__.py)
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_OCR2_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===================
    # Model Configuration
    # ===================
    model_path: str = Field(
        default="deepseek-ai/DeepSeek-OCR-2",
        description="Path to the model (HuggingFace model ID or local path)"
    )
    dtype: str = Field(
        default="bfloat16",
        description="Data type for model weights (bfloat16, float16, float32)"
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code from HuggingFace"
    )

    # =================
    # GPU Configuration
    # =================
    cuda_visible_devices: str = Field(
        default="0",
        description="CUDA visible devices (e.g., '0', '0,1', '0,1,2,3')"
    )
    gpu_memory_utilization: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization ratio (0.1-1.0)"
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism"
    )

    # ==================
    # vLLM Configuration
    # ==================
    max_model_len: int = Field(
        default=8192,
        ge=1024,
        description="Maximum sequence length for the model"
    )
    block_size: int = Field(
        default=256,
        description="KV cache block size"
    )
    swap_space: int = Field(
        default=0,
        ge=0,
        description="CPU swap space in GB"
    )
    max_num_seqs: int = Field(
        default=100,
        ge=1,
        description="Maximum number of concurrent sequences"
    )
    enforce_eager: bool = Field(
        default=False,
        description="Whether to enforce eager mode (disable CUDA graphs)"
    )
    disable_mm_preprocessor_cache: bool = Field(
        default=False,
        description="Whether to disable multimodal preprocessor cache"
    )

    # ==========================
    # Image Processing Settings
    # ==========================
    image_size: int = Field(
        default=768,
        description="Local view image size for dynamic cropping"
    )
    base_size: int = Field(
        default=1024,
        description="Global view image size"
    )
    min_crops: int = Field(
        default=2,
        ge=1,
        description="Minimum number of crops for dynamic preprocessing"
    )
    max_crops: int = Field(
        default=6,
        ge=1,
        description="Maximum number of crops for dynamic preprocessing"
    )
    crop_mode: bool = Field(
        default=True,
        description="Whether to enable dynamic cropping"
    )
    num_workers: int = Field(
        default=64,
        ge=1,
        description="Number of workers for image preprocessing"
    )

    # ===================
    # Sampling Parameters
    # ===================
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 for deterministic)"
    )
    max_tokens: int = Field(
        default=8192,
        ge=1,
        description="Maximum number of tokens to generate"
    )
    ngram_size: int = Field(
        default=20,
        ge=1,
        description="N-gram size for repetition penalty"
    )
    window_size: int = Field(
        default=90,
        ge=1,
        description="Window size for N-gram repetition check"
    )
    skip_special_tokens: bool = Field(
        default=False,
        description="Whether to skip special tokens in output"
    )

    # ================
    # PDF Settings
    # ================
    pdf_dpi: int = Field(
        default=144,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion"
    )
    pdf_image_format: str = Field(
        default="PNG",
        description="Image format for PDF conversion (PNG, JPEG)"
    )
    page_separator: str = Field(
        default="\n<--- Page Split --->\n",
        description="Separator between PDF pages in output"
    )
    skip_repeat_pages: bool = Field(
        default=True,
        description="Whether to skip repeated pages in PDF"
    )

    # ==============
    # Default Prompt
    # ==============
    default_prompt: str = Field(
        default="<image>\n<|grounding|>Convert the document to markdown.",
        description="Default prompt for OCR"
    )

    # ====================
    # Server Configuration
    # ====================
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of uvicorn workers"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )

    # ============
    # Engine Mode
    # ============
    engine_mode: Literal["sync", "async"] = Field(
        default="sync",
        description="Engine mode: 'sync' for LLM, 'async' for AsyncLLMEngine"
    )

    # =================
    # Output Settings
    # =================
    output_dir: str = Field(
        default="/tmp/deepseek_ocr2_output",
        description="Directory for temporary output files"
    )
    cleanup_temp_files: bool = Field(
        default=True,
        description="Whether to cleanup temporary files after processing"
    )

    # ==================
    # Whitelist Tokens
    # ==================
    whitelist_token_ids: str = Field(
        default="128821,128822",
        description="Comma-separated whitelist token IDs for N-gram processor"
    )

    @property
    def whitelist_token_ids_set(self) -> set:
        """Parse whitelist token IDs to a set."""
        if not self.whitelist_token_ids:
            return set()
        return {int(x.strip()) for x in self.whitelist_token_ids.split(",") if x.strip()}

    def get_vllm_env_vars(self) -> dict:
        """Get environment variables for vLLM."""
        return {
            "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
            "VLLM_USE_V1": "0",
        }

    def apply_env_vars(self):
        """Apply environment variables before engine initialization."""
        for key, value in self.get_vllm_env_vars().items():
            os.environ[key] = value


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global _settings
    current = get_settings().model_dump()
    current.update(kwargs)
    _settings = Settings(**current)
    return _settings


def reset_settings():
    """Reset settings to default."""
    global _settings
    _settings = None
