#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  LiteAgent Installer
#  Usage: curl -sL <url>/install.sh | bash
#     or: ./install.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║   🤖  LiteAgent v1.0.0 — Installer               ║
    ║                                                   ║
    ║   AI Agent with Memory + 8 Metacognition Features ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# ── Detect Python ────────────────────────────────────────────────
find_python() {
    local candidates=("python3.12" "python3.11" "python3.10" "python3")
    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# ── Main ─────────────────────────────────────────────────────────
main() {
    banner

    # 1. Check Python
    info "Checking Python version..."
    PYTHON=$(find_python) || {
        err "Python 3.10+ is required but not found."
        echo ""
        echo "  Install options:"
        echo "    macOS:   brew install python@3.11"
        echo "    Ubuntu:  sudo apt install python3.11 python3.11-venv"
        echo "    Fedora:  sudo dnf install python3.11"
        echo ""
        exit 1
    }
    PYTHON_VER=$("$PYTHON" --version 2>&1)
    ok "Found $PYTHON_VER ($PYTHON)"

    # 2. Create virtual environment
    VENV_DIR="${LITEAGENT_VENV:-.venv}"
    if [[ -d "$VENV_DIR" ]]; then
        warn "Virtual environment $VENV_DIR already exists, reusing."
    else
        info "Creating virtual environment in $VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR"
        ok "Virtual environment created."
    fi

    # Activate
    source "$VENV_DIR/bin/activate"

    # 3. Upgrade pip
    info "Upgrading pip..."
    pip install --quiet --upgrade pip wheel setuptools
    ok "pip upgraded."

    # 4. Detect installation mode
    echo ""
    echo -e "${BOLD}Select installation mode:${NC}"
    echo ""
    echo "  1) ${GREEN}Minimal${NC}     — CLI only (anthropic SDK)"
    echo "  2) ${CYAN}Standard${NC}    — CLI + API/Dashboard + Embeddings"
    echo "  3) ${YELLOW}Full${NC}        — Everything (+ Telegram bot + dev tools)"
    echo "  4) ${RED}Custom${NC}      — Choose components"
    echo ""

    read -r -p "Your choice [1-4, default=2]: " CHOICE
    CHOICE="${CHOICE:-2}"

    case "$CHOICE" in
        1)
            info "Installing minimal (CLI only)..."
            pip install --quiet -e .
            ;;
        2)
            info "Installing standard (API + Embeddings)..."
            pip install --quiet -e ".[api,embeddings]"
            ;;
        3)
            info "Installing full (all components)..."
            pip install --quiet -e ".[all]"
            ;;
        4)
            EXTRAS=""
            read -r -p "  API/Dashboard? [Y/n]: " ans
            [[ "${ans:-y}" =~ ^[Yy] ]] && EXTRAS="${EXTRAS}api,"

            read -r -p "  Telegram bot? [y/N]: " ans
            [[ "${ans:-n}" =~ ^[Yy] ]] && EXTRAS="${EXTRAS}telegram,"

            read -r -p "  Embeddings (semantic memory)? [Y/n]: " ans
            [[ "${ans:-y}" =~ ^[Yy] ]] && EXTRAS="${EXTRAS}embeddings,"

            read -r -p "  Dev tools (pytest)? [y/N]: " ans
            [[ "${ans:-n}" =~ ^[Yy] ]] && EXTRAS="${EXTRAS}dev,"

            # Remove trailing comma
            EXTRAS="${EXTRAS%,}"
            if [[ -n "$EXTRAS" ]]; then
                info "Installing with [$EXTRAS]..."
                pip install --quiet -e ".[$EXTRAS]"
            else
                info "Installing minimal..."
                pip install --quiet -e .
            fi
            ;;
        *)
            err "Invalid choice. Exiting."
            exit 1
            ;;
    esac
    ok "Dependencies installed."

    # 5. Setup API key
    echo ""
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        warn "ANTHROPIC_API_KEY is not set."
        echo ""
        read -r -p "Enter your Anthropic API key (or press Enter to skip): " API_KEY
        if [[ -n "$API_KEY" ]]; then
            # Save to .env
            echo "ANTHROPIC_API_KEY=$API_KEY" > .env
            ok "API key saved to .env"
            echo ""
            echo -e "  ${YELLOW}Add to your shell profile:${NC}"
            echo "  export ANTHROPIC_API_KEY=$API_KEY"
        else
            warn "Skipped. Set ANTHROPIC_API_KEY before running LiteAgent."
        fi
    else
        ok "ANTHROPIC_API_KEY is already set."
    fi

    # 6. Create default config if missing
    if [[ ! -f "config.json" ]]; then
        info "Creating default config.json..."
        cat > config.json << 'CONF'
{
  "agent": {
    "name": "LiteAgent",
    "soul": "soul.md",
    "max_iterations": 15,
    "default_model": "claude-sonnet-4-20250514",
    "models": {
      "simple": "claude-haiku-4-5-20251001",
      "medium": "claude-sonnet-4-20250514",
      "complex": "claude-opus-4-20250115"
    }
  },
  "memory": {
    "db_path": "~/.liteagent/memory.db",
    "max_history_tokens": 2000,
    "keep_recent_messages": 6,
    "auto_learn": true,
    "extraction_model": "claude-haiku-4-5-20251001"
  },
  "tools": {
    "builtin": ["read_file", "write_file", "exec_command", "web_search"],
    "mcp_servers": {}
  },
  "channels": {
    "cli": { "enabled": true },
    "telegram": { "enabled": false, "token_env": "TELEGRAM_BOT_TOKEN" },
    "api": { "enabled": false, "host": "0.0.0.0", "port": 8080 }
  },
  "cost": {
    "cascade_routing": true,
    "prompt_caching": true,
    "context_compression": true,
    "budget_daily_usd": 5.0,
    "track_usage": true
  },
  "features": {
    "dream_cycle":            { "enabled": false, "cron": "0 3 * * *", "similarity_threshold": 0.85, "decay_rate": 0.05, "max_consolidations_per_run": 20 },
    "self_evolving_prompt":   { "enabled": false, "min_friction_signals": 5, "review_cron": "0 4 * * 0", "auto_apply": false },
    "proactive_agent":        { "enabled": false, "pattern_window_days": 30, "min_pattern_occurrences": 3 },
    "auto_tool_synthesis":    { "enabled": false, "auto_approve": false },
    "confidence_gate":        { "enabled": false, "threshold": 6, "escalate_to_model": true },
    "style_adaptation":       { "enabled": false, "ema_alpha": 0.3 },
    "skill_crystallization":  { "enabled": false, "min_tool_calls": 3 },
    "counterfactual_replay":  { "enabled": false, "cron": "0 4 * * *", "max_replays_per_run": 10 }
  }
}
CONF
        ok "config.json created."
    else
        ok "config.json already exists."
    fi

    # 7. Create data directory
    mkdir -p ~/.liteagent
    ok "Data directory ~/.liteagent ready."

    # 8. Verify installation
    echo ""
    info "Verifying installation..."
    if "$VENV_DIR/bin/python" -c "from liteagent.agent import LiteAgent; print('LiteAgent imported OK')" 2>/dev/null; then
        ok "LiteAgent installed successfully!"
    else
        err "Import check failed. Try: pip install -e ."
        exit 1
    fi

    # 9. Summary
    echo ""
    echo -e "${GREEN}${BOLD}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  Installation complete!${NC}"
    echo -e "${GREEN}${BOLD}═══════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${BOLD}Quick start:${NC}"
    echo ""
    echo -e "    ${CYAN}source $VENV_DIR/bin/activate${NC}"
    echo -e "    ${CYAN}liteagent${NC}                        # interactive CLI"
    echo -e "    ${CYAN}liteagent -1 \"Hello!\"${NC}             # one-shot mode"
    echo ""
    echo -e "  ${BOLD}API + Dashboard:${NC}"
    echo ""
    echo -e "    ${CYAN}liteagent --channel api${NC}           # start on :8080"
    echo -e "    Open ${YELLOW}http://localhost:8080${NC}       # dashboard"
    echo ""
    echo -e "  ${BOLD}Useful commands:${NC}"
    echo ""
    echo -e "    ${CYAN}make test${NC}                         # run 143 tests"
    echo -e "    ${CYAN}make run${NC}                          # start CLI"
    echo -e "    ${CYAN}make api${NC}                          # start API server"
    echo -e "    ${CYAN}make docker${NC}                       # build Docker image"
    echo ""
    echo -e "  ${BOLD}Configuration:${NC}  ${YELLOW}config.json${NC}"
    echo -e "  ${BOLD}Documentation:${NC}  ${YELLOW}README.md${NC}"
    echo ""
}

main "$@"
