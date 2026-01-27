#!/bin/bash
#
# DeepSeek-OCR-2 API Server Startup Script
#
# Usage:
#   ./start.sh [options]
#
# Examples:
#   ./start.sh                                    # Start with defaults
#   ./start.sh --gpu-memory-utilization 0.9      # Use 90% GPU memory
#   ./start.sh --port 8080 --cuda-devices 0,1    # Custom port and GPUs
#

set -e

# Default values
HOST="${DEEPSEEK_OCR2_HOST:-0.0.0.0}"
PORT="${DEEPSEEK_OCR2_PORT:-8000}"
WORKERS="${DEEPSEEK_OCR2_WORKERS:-1}"
MODEL_PATH="${DEEPSEEK_OCR2_MODEL_PATH:-deepseek-ai/DeepSeek-OCR-2}"
DTYPE="${DEEPSEEK_OCR2_DTYPE:-bfloat16}"
CUDA_DEVICES="${DEEPSEEK_OCR2_CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY="${DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION:-0.8}"
TENSOR_PARALLEL="${DEEPSEEK_OCR2_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${DEEPSEEK_OCR2_MAX_MODEL_LEN:-8192}"
ENGINE_MODE="${DEEPSEEK_OCR2_ENGINE_MODE:-sync}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --engine-mode)
            ENGINE_MODE="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --help|-h)
            echo "DeepSeek-OCR-2 API Server"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Server Options:"
            echo "  --host HOST                    Server host (default: 0.0.0.0)"
            echo "  --port PORT                    Server port (default: 8000)"
            echo "  --workers N                    Number of workers (default: 1)"
            echo "  --reload                       Enable auto-reload"
            echo ""
            echo "Model Options:"
            echo "  --model-path PATH              Model path (default: deepseek-ai/DeepSeek-OCR-2)"
            echo "  --dtype TYPE                   Data type: bfloat16, float16, float32 (default: bfloat16)"
            echo ""
            echo "GPU Options:"
            echo "  --cuda-devices DEVICES         CUDA devices (default: 0)"
            echo "  --gpu-memory-utilization RATIO GPU memory ratio 0.1-1.0 (default: 0.8)"
            echo "  --tensor-parallel-size N       Tensor parallel GPUs (default: 1)"
            echo ""
            echo "vLLM Options:"
            echo "  --max-model-len N              Max sequence length (default: 8192)"
            echo "  --engine-mode MODE             sync or async (default: sync)"
            echo ""
            echo "Environment Variables:"
            echo "  All options can be set via DEEPSEEK_OCR2_* environment variables"
            echo "  Example: DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION=0.9"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Export environment variables
export DEEPSEEK_OCR2_HOST="$HOST"
export DEEPSEEK_OCR2_PORT="$PORT"
export DEEPSEEK_OCR2_WORKERS="$WORKERS"
export DEEPSEEK_OCR2_MODEL_PATH="$MODEL_PATH"
export DEEPSEEK_OCR2_DTYPE="$DTYPE"
export DEEPSEEK_OCR2_CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION="$GPU_MEMORY"
export DEEPSEEK_OCR2_TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL"
export DEEPSEEK_OCR2_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export DEEPSEEK_OCR2_ENGINE_MODE="$ENGINE_MODE"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Disable vLLM V1 engine
export VLLM_USE_V1=0

# Print configuration
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           DeepSeek-OCR-2 API Server                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:     $MODEL_PATH"
echo "║  Host:      $HOST"
echo "║  Port:      $PORT"
echo "║  Workers:   $WORKERS"
echo "║  GPU Mem:   $GPU_MEMORY"
echo "║  CUDA:      $CUDA_DEVICES"
echo "║  Mode:      $ENGINE_MODE"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Docs:      http://$HOST:$PORT/docs"
echo "║  ReDoc:     http://$HOST:$PORT/redoc"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR/.."

# Start the server
if [ -n "$RELOAD" ]; then
    python -m deepseek_ocr2_api \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --model-path "$MODEL_PATH" \
        --dtype "$DTYPE" \
        --cuda-devices "$CUDA_DEVICES" \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --engine-mode "$ENGINE_MODE" \
        --reload
else
    python -m deepseek_ocr2_api \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --model-path "$MODEL_PATH" \
        --dtype "$DTYPE" \
        --cuda-devices "$CUDA_DEVICES" \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --engine-mode "$ENGINE_MODE"
fi
