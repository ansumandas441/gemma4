#!/bin/bash

# Unified management script for Gemma 4 models
# Usage: ./manage_models.sh {31b|26b|all} {start|stop|restart|status|set-key <KEY>|remove-key|logs}

set -e

get_config() {
    local model=$1 field=$2
    case "${model}_${field}" in
        31b_script) echo "/home/primetrace/start_gemma4.sh" ;;
        26b_script) echo "/home/primetrace/start_gemma4_26b.sh" ;;
        31b_tmux)   echo "gemma4" ;;
        26b_tmux)   echo "gemma4-26b" ;;
        31b_log)    echo "/home/primetrace/gemma4-serve.log" ;;
        26b_log)    echo "/home/primetrace/gemma4-26b-serve.log" ;;
        31b_port)   echo "8000" ;;
        26b_port)   echo "8010" ;;
        31b_name)   echo "Gemma 4 31B-it" ;;
        26b_name)   echo "Gemma 4 26B-A4B" ;;
    esac
}

usage() {
    echo "Usage: $0 {31b|26b|all} {start|stop|restart|status|set-key <KEY>|remove-key|logs}"
    echo ""
    echo "Models:"
    echo "  31b   - Gemma 4 31B-it   (GPUs 0,1 | port 8000)"
    echo "  26b   - Gemma 4 26B-A4B  (GPUs 2,3 | port 9835)"
    echo "  all   - Both models"
    echo ""
    echo "Actions:"
    echo "  start      - Start the model server"
    echo "  stop       - Stop the model server"
    echo "  restart    - Restart the model server"
    echo "  status     - Show running status"
    echo "  set-key K  - Set API key to K and restart"
    echo "  remove-key - Remove API key and restart"
    echo "  logs       - Tail the log file (Ctrl+C to exit)"
    exit 1
}

do_start() {
    local model=$1
    local tmux_name=$(get_config "$model" tmux)
    local script=$(get_config "$model" script)
    local log=$(get_config "$model" log)
    local name=$(get_config "$model" name)
    local port=$(get_config "$model" port)

    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        echo "[$name] Already running in tmux session '$tmux_name'"
        return
    fi
    echo "[$name] Starting on port $port..."
    tmux new-session -d -s "$tmux_name" "$script 2>&1 | tee $log"
    echo "[$name] Started. Takes ~90s to load. Check: $0 $model logs"
}

do_stop() {
    local model=$1
    local tmux_name=$(get_config "$model" tmux)
    local name=$(get_config "$model" name)

    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        tmux kill-session -t "$tmux_name"
        echo "[$name] Stopped."
    else
        echo "[$name] Not running."
    fi
}

do_status() {
    local model=$1
    local tmux_name=$(get_config "$model" tmux)
    local port=$(get_config "$model" port)
    local name=$(get_config "$model" name)
    local script=$(get_config "$model" script)

    echo "=== $name ==="
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        echo "  Status:  RUNNING (tmux: $tmux_name)"
    else
        echo "  Status:  STOPPED"
    fi
    echo "  Port:    $port"

    if grep -q '\-\-api-key' "$script" 2>/dev/null; then
        echo "  API Key: Enabled"
    else
        echo "  API Key: Disabled (open access)"
    fi

    local key=$(grep -oP '(?<=--api-key )\S+' "$script" 2>/dev/null || true)
    local header=""
    [ -n "$key" ] && header="Authorization: Bearer $key"
    if [ -n "$header" ]; then
        resp=$(curl -s --max-time 2 -H "$header" "http://localhost:$port/v1/models" 2>/dev/null || true)
    else
        resp=$(curl -s --max-time 2 "http://localhost:$port/v1/models" 2>/dev/null || true)
    fi
    if echo "$resp" | grep -q '"object"' 2>/dev/null; then
        echo "  API:     Responding"
    else
        echo "  API:     Not responding (loading or stopped)"
    fi
    echo ""
}

do_set_key() {
    local model=$1
    local key=$2
    local script=$(get_config "$model" script)
    local name=$(get_config "$model" name)

    if [ -z "$key" ]; then
        echo "Error: No key provided. Usage: $0 $model set-key YOUR_KEY"
        exit 1
    fi

    sed -i '/--api-key/d' "$script"
    sed -i "s|--host|--api-key $key \\\\\n  --host|" "$script"
    echo "[$name] API key updated. Restarting..."
    do_stop "$model"
    sleep 2
    do_start "$model"
}

do_remove_key() {
    local model=$1
    local script=$(get_config "$model" script)
    local name=$(get_config "$model" name)

    sed -i '/--api-key/d' "$script"
    echo "[$name] API key removed. Restarting..."
    do_stop "$model"
    sleep 2
    do_start "$model"
}

do_logs() {
    local model=$1
    local log=$(get_config "$model" log)
    local name=$(get_config "$model" name)

    echo "[$name] Tailing $log (Ctrl+C to exit)..."
    tail -f "$log"
}

# --- Main ---
[ $# -lt 2 ] && usage

MODEL=$1
ACTION=$2
shift 2

if [ "$MODEL" = "all" ]; then
    for m in 31b 26b; do
        case $ACTION in
            start)      do_start "$m" ;;
            stop)       do_stop "$m" ;;
            restart)    do_stop "$m"; sleep 2; do_start "$m" ;;
            status)     do_status "$m" ;;
            set-key)    do_set_key "$m" "$1" ;;
            remove-key) do_remove_key "$m" ;;
            logs)       echo "Cannot tail logs for 'all'. Use: $0 31b logs  or  $0 26b logs"; exit 1 ;;
            *)          usage ;;
        esac
    done
else
    case $MODEL in
        31b|26b) ;;
        *) usage ;;
    esac
    case $ACTION in
        start)      do_start "$MODEL" ;;
        stop)       do_stop "$MODEL" ;;
        restart)    do_stop "$MODEL"; sleep 2; do_start "$MODEL" ;;
        status)     do_status "$MODEL" ;;
        set-key)    do_set_key "$MODEL" "$1" ;;
        remove-key) do_remove_key "$MODEL" ;;
        logs)       do_logs "$MODEL" ;;
        *)          usage ;;
    esac
fi
