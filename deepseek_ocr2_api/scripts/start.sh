#!/bin/bash
#
# DeepSeek-OCR-2 API Server Startup Script
#
# Configuration priority (highest to lowest):
#   1. Command line arguments
#   2. Environment variables (DEEPSEEK_OCR_*)
#   3. .env file
#   4. Default values in config.py
#
# Usage:
#   ./start.sh [options]
#
# Examples:
#   ./start.sh                                    # Start with defaults/.env
#   ./start.sh --gpu-memory-utilization 0.9      # Override GPU memory
#   ./start.sh --env-file /path/to/.env          # Use custom .env file
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

# Default .env file location
ENV_FILE="${ROOT_DIR}/.env"

# Collect CLI arguments to pass through
CLI_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --help|-h)
            cat << 'EOF'
DeepSeek-OCR-2 API Server

Usage: ./start.sh [options]

Configuration is loaded from (highest priority first):
  1. Command line arguments (passed to this script)
  2. Environment variables (DEEPSEEK_OCR_*)
  3. .env file (default: project root, or specify with --env-file)
  4. Default values in config.py

Script Options:
  --env-file PATH        Path to .env file (default: project root/.env)
  --help, -h             Show this help message

All other options are passed directly to the Python server.
Run 'python -m deepseek_ocr2_api --help' for full server options.

Common Options:
  --host HOST                    Server host (default: 0.0.0.0)
  --port PORT                    Server port (default: 8000)
  --model-path PATH              Model path
  --gpu-memory-utilization RATIO GPU memory ratio 0.1-1.0
  --tensor-parallel-size N       Tensor parallel GPUs
  --mode MODE                    Engine mode: sync or async
  --reload                       Enable auto-reload for development

Environment Variables (prefix: DEEPSEEK_OCR_):
  DEEPSEEK_OCR_MODEL_PATH
  DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION
  DEEPSEEK_OCR_TENSOR_PARALLEL_SIZE
  DEEPSEEK_OCR_HOST
  DEEPSEEK_OCR_PORT
  ... (see config.py for full list)

Examples:
  # Use .env file configuration
  ./start.sh

  # Override specific settings
  ./start.sh --gpu-memory-utilization 0.8 --port 8080

  # Use custom .env file
  ./start.sh --env-file /path/to/custom.env

  # Development mode with auto-reload
  ./start.sh --reload
EOF
            exit 0
            ;;
        *)
            # Pass through all other arguments
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load .env file if it exists (without overriding existing env vars)
if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from: $ENV_FILE"
    # Export variables from .env only if not already set
    set -a
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        # Only set if not already in environment
        if [ -z "${!key}" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
    set +a
fi

# Set required environment variables for vLLM
export VLLM_USE_V1=0

# Set CUDA_VISIBLE_DEVICES if specified in env
if [ -n "$DEEPSEEK_OCR_CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$DEEPSEEK_OCR_CUDA_VISIBLE_DEVICES"
fi

# Print startup banner
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           DeepSeek-OCR-2 API Server                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Change to project root directory
cd "$ROOT_DIR"

# Start the server with any CLI arguments
exec python -m deepseek_ocr2_api "${CLI_ARGS[@]}"
