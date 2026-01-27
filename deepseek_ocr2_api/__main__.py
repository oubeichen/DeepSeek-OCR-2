"""
DeepSeek-OCR-2 API Server - Command Line Entry Point

Supports running with: python -m deepseek_ocr2_api
"""

import argparse
import os
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR-2 API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server settings
    server_group = parser.add_argument_group("Server Settings")
    server_group.add_argument(
        "--host",
        type=str,
        default=os.getenv("DEEPSEEK_OCR2_HOST", "0.0.0.0"),
        help="Server host address",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_PORT", "8000")),
        help="Server port",
    )
    server_group.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_WORKERS", "1")),
        help="Number of uvicorn workers",
    )
    server_group.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("DEEPSEEK_OCR2_MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2"),
        help="Path to the model (HuggingFace ID or local path)",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default=os.getenv("DEEPSEEK_OCR2_DTYPE", "bfloat16"),
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights",
    )

    # GPU settings
    gpu_group = parser.add_argument_group("GPU Settings")
    gpu_group.add_argument(
        "--cuda-devices",
        type=str,
        default=os.getenv("DEEPSEEK_OCR2_CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA visible devices (e.g., '0', '0,1')",
    )
    gpu_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION", "0.9")),
        help="GPU memory utilization ratio (0.1-1.0)",
    )
    gpu_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_TENSOR_PARALLEL_SIZE", "1")),
        help="Number of GPUs for tensor parallelism",
    )

    # vLLM settings
    vllm_group = parser.add_argument_group("vLLM Settings")
    vllm_group.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_MAX_MODEL_LEN", "8192")),
        help="Maximum sequence length",
    )
    vllm_group.add_argument(
        "--block-size",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_BLOCK_SIZE", "256")),
        help="KV cache block size",
    )
    vllm_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_MAX_NUM_SEQS", "100")),
        help="Maximum number of concurrent sequences",
    )
    vllm_group.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graphs)",
    )

    # Engine mode
    engine_group = parser.add_argument_group("Engine Settings")
    engine_group.add_argument(
        "--engine-mode",
        type=str,
        default=os.getenv("DEEPSEEK_OCR2_ENGINE_MODE", "sync"),
        choices=["sync", "async"],
        help="Engine mode: sync for LLM, async for AsyncLLMEngine",
    )

    # Image processing settings
    image_group = parser.add_argument_group("Image Processing Settings")
    image_group.add_argument(
        "--image-size",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_IMAGE_SIZE", "768")),
        help="Local view image size",
    )
    image_group.add_argument(
        "--base-size",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_BASE_SIZE", "1024")),
        help="Global view image size",
    )
    image_group.add_argument(
        "--min-crops",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_MIN_CROPS", "2")),
        help="Minimum number of crops",
    )
    image_group.add_argument(
        "--max-crops",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_MAX_CROPS", "6")),
        help="Maximum number of crops",
    )
    image_group.add_argument(
        "--no-crop-mode",
        action="store_true",
        help="Disable dynamic cropping",
    )

    # Sampling settings
    sampling_group = parser.add_argument_group("Sampling Settings")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("DEEPSEEK_OCR2_TEMPERATURE", "0.0")),
        help="Sampling temperature",
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_MAX_TOKENS", "8192")),
        help="Maximum tokens to generate",
    )
    sampling_group.add_argument(
        "--ngram-size",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_NGRAM_SIZE", "20")),
        help="N-gram size for repetition penalty",
    )
    sampling_group.add_argument(
        "--window-size",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_WINDOW_SIZE", "90")),
        help="Window size for N-gram check",
    )

    # PDF settings
    pdf_group = parser.add_argument_group("PDF Settings")
    pdf_group.add_argument(
        "--pdf-dpi",
        type=int,
        default=int(os.getenv("DEEPSEEK_OCR2_PDF_DPI", "144")),
        help="DPI for PDF to image conversion",
    )

    # Prompt
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=os.getenv("DEEPSEEK_OCR2_DEFAULT_PROMPT", "<image>\n<|grounding|>Convert the document to markdown."),
        help="Default prompt for OCR",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set environment variables from arguments
    env_mapping = {
        "DEEPSEEK_OCR2_HOST": args.host,
        "DEEPSEEK_OCR2_PORT": str(args.port),
        "DEEPSEEK_OCR2_WORKERS": str(args.workers),
        "DEEPSEEK_OCR2_MODEL_PATH": args.model_path,
        "DEEPSEEK_OCR2_DTYPE": args.dtype,
        "DEEPSEEK_OCR2_CUDA_VISIBLE_DEVICES": args.cuda_devices,
        "DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        "DEEPSEEK_OCR2_TENSOR_PARALLEL_SIZE": str(args.tensor_parallel_size),
        "DEEPSEEK_OCR2_MAX_MODEL_LEN": str(args.max_model_len),
        "DEEPSEEK_OCR2_BLOCK_SIZE": str(args.block_size),
        "DEEPSEEK_OCR2_MAX_NUM_SEQS": str(args.max_num_seqs),
        "DEEPSEEK_OCR2_ENFORCE_EAGER": str(args.enforce_eager).lower(),
        "DEEPSEEK_OCR2_ENGINE_MODE": args.engine_mode,
        "DEEPSEEK_OCR2_IMAGE_SIZE": str(args.image_size),
        "DEEPSEEK_OCR2_BASE_SIZE": str(args.base_size),
        "DEEPSEEK_OCR2_MIN_CROPS": str(args.min_crops),
        "DEEPSEEK_OCR2_MAX_CROPS": str(args.max_crops),
        "DEEPSEEK_OCR2_CROP_MODE": str(not args.no_crop_mode).lower(),
        "DEEPSEEK_OCR2_TEMPERATURE": str(args.temperature),
        "DEEPSEEK_OCR2_MAX_TOKENS": str(args.max_tokens),
        "DEEPSEEK_OCR2_NGRAM_SIZE": str(args.ngram_size),
        "DEEPSEEK_OCR2_WINDOW_SIZE": str(args.window_size),
        "DEEPSEEK_OCR2_PDF_DPI": str(args.pdf_dpi),
        "DEEPSEEK_OCR2_DEFAULT_PROMPT": args.default_prompt,
    }

    for key, value in env_mapping.items():
        # Only set if not already in environment (preserves .env file values)
        if key not in os.environ:
            os.environ[key] = value

    # Import uvicorn here to avoid import issues
    import uvicorn

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           DeepSeek-OCR-2 API Server                          ║
╠══════════════════════════════════════════════════════════════╣
║  Model:     {args.model_path:<47} ║
║  Host:      {args.host:<47} ║
║  Port:      {args.port:<47} ║
║  GPU Mem:   {args.gpu_memory_utilization:<47} ║
║  Mode:      {args.engine_mode:<47} ║
╠══════════════════════════════════════════════════════════════╣
║  Docs:      http://{args.host}:{args.port}/docs{' ' * (36 - len(str(args.port)))} ║
║  ReDoc:     http://{args.host}:{args.port}/redoc{' ' * (35 - len(str(args.port)))} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "deepseek_ocr2_api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
