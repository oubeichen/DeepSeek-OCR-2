#!/bin/bash
#
# DeepSeek-OCR-2 API Server Control Script
#
# Usage:
#   ./ctl.sh {start|stop|restart|status} [options]
#
# Commands:
#   start   - Start the server in background
#   stop    - Stop the server gracefully
#   restart - Restart the server
#   status  - Check server status
#
# All options after the command are passed to start.sh
#
# Examples:
#   ./ctl.sh start
#   ./ctl.sh start --gpu-memory-utilization 0.9
#   ./ctl.sh stop
#   ./ctl.sh restart
#   ./ctl.sh status
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

# Service name and PID file location
SERVICE_NAME="deepseek-ocr2-api"
PID_DIR="${ROOT_DIR}/run"
PID_FILE="${PID_DIR}/${SERVICE_NAME}.pid"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${SERVICE_NAME}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "$PID_DIR" "$LOG_DIR"

# Print colored message
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Get PID from file
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

# Check if process is running
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Start the server
do_start() {
    if is_running; then
        local pid=$(get_pid)
        print_status "$YELLOW" "Server is already running (PID: $pid)"
        return 1
    fi

    print_status "$GREEN" "Starting ${SERVICE_NAME}..."

    # Remove stale PID file if exists
    rm -f "$PID_FILE"

    # Start the server in background, redirecting output to log file
    nohup "$SCRIPT_DIR/start.sh" "$@" >> "$LOG_FILE" 2>&1 &
    local pid=$!

    # Save PID
    echo "$pid" > "$PID_FILE"

    # Wait a moment and check if it started successfully
    sleep 2

    if is_running; then
        print_status "$GREEN" "Server started successfully (PID: $pid)"
        print_status "$GREEN" "Log file: $LOG_FILE"
        return 0
    else
        print_status "$RED" "Failed to start server. Check log file: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Stop the server
do_stop() {
    if ! is_running; then
        print_status "$YELLOW" "Server is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(get_pid)
    print_status "$GREEN" "Stopping ${SERVICE_NAME} (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null

    # Wait for process to stop (max 30 seconds)
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        print_status "$YELLOW" "Process did not stop gracefully, sending SIGKILL..."
        kill -KILL "$pid" 2>/dev/null
        sleep 1
    fi

    rm -f "$PID_FILE"

    if kill -0 "$pid" 2>/dev/null; then
        print_status "$RED" "Failed to stop server"
        return 1
    else
        print_status "$GREEN" "Server stopped successfully"
        return 0
    fi
}

# Restart the server
do_restart() {
    do_stop
    sleep 2
    do_start "$@"
}

# Show server status
do_status() {
    if is_running; then
        local pid=$(get_pid)
        print_status "$GREEN" "Server is running (PID: $pid)"

        # Show process info
        echo ""
        echo "Process details:"
        ps -p "$pid" -o pid,ppid,user,%cpu,%mem,etime,command 2>/dev/null || true

        # Show last few lines of log
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Recent log entries:"
            tail -5 "$LOG_FILE" 2>/dev/null || true
        fi
        return 0
    else
        print_status "$RED" "Server is not running"
        if [ -f "$PID_FILE" ]; then
            print_status "$YELLOW" "Stale PID file found, removing..."
            rm -f "$PID_FILE"
        fi
        return 1
    fi
}

# Show help
show_help() {
    cat << 'EOF'
DeepSeek-OCR-2 API Server Control Script

Usage: ./ctl.sh {start|stop|restart|status} [options]

Commands:
  start     Start the server in background
  stop      Stop the server gracefully
  restart   Restart the server
  status    Check server status

Options after command are passed to start.sh (for start/restart):
  --env-file PATH              Path to .env file
  --host HOST                  Server host
  --port PORT                  Server port
  --gpu-memory-utilization N   GPU memory ratio 0.1-1.0
  ... (see start.sh --help for full options)

Files:
  PID file: ${PID_FILE}
  Log file: ${LOG_FILE}

Examples:
  ./ctl.sh start                              # Start with defaults
  ./ctl.sh start --port 8080                  # Start on port 8080
  ./ctl.sh stop                               # Stop gracefully
  ./ctl.sh restart --gpu-memory-utilization 0.9
  ./ctl.sh status                             # Check if running

For systemd integration, see: install-service.sh
EOF
}

# Main entry point
case "${1:-}" in
    start)
        shift
        do_start "$@"
        ;;
    stop)
        do_stop
        ;;
    restart)
        shift
        do_restart "$@"
        ;;
    status)
        do_status
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|help}"
        echo "Run '$0 help' for more information"
        exit 1
        ;;
esac
