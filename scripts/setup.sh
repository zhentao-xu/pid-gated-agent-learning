#!/usr/bin/env bash
# Interactive TUI setup for pid-gated-agent-learning.
#
# Walks a first-time user through:
#   1. Platform detection (macOS / Linux / WSL — Windows gets manual guidance)
#   2. Installing uv if missing (offers brew / curl / wget / pip depending on
#      what's actually available on the user's system — no assumptions)
#   3. Picking a Python version
#   4. Picking which LLM providers to configure
#   5. Collecting API keys (hidden input) and writing .env
#   6. Running `uv sync` to set up the dev environment
#   7. (Optional) launching the quickstart Jupyter notebook
#
# Usage: ./scripts/setup.sh
# Requires: bash 4+ (bundled with macOS and all mainstream Linux distros).

set -u

# ---- Styling --------------------------------------------------------------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
  BOLD="$(tput bold)";  DIM="$(tput dim)";   RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; MAGENTA="$(tput setaf 5)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

# All TUI output is written to stderr so it stays visible even when a helper
# function is invoked inside $(...) command substitution (which captures stdout).
banner()   { printf '\n%s%s%s\n' "$BOLD$MAGENTA" "$1" "$RESET" >&2; printf '%s%s%s\n\n' "$DIM" "${1//?/─}" "$RESET" >&2; }
step()     { printf '\n%s▸ %s%s\n' "$BOLD$BLUE" "$1" "$RESET" >&2; }
info()     { printf '  %s%s%s\n' "$DIM" "$1" "$RESET" >&2; }
ok()       { printf '  %s✓ %s%s\n' "$GREEN" "$1" "$RESET" >&2; }
warn()     { printf '  %s! %s%s\n' "$YELLOW" "$1" "$RESET" >&2; }
err()      { printf '  %s✗ %s%s\n' "$RED" "$1" "$RESET" >&2; }
prompt()   { printf '%s? %s%s ' "$BOLD$CYAN" "$1" "$RESET" >&2; }

ask_yn() {  # ask_yn "Prompt" default_yes_or_no → returns 0 for yes, 1 for no
  local p="$1"; local default="${2:-y}"; local hint; local reply
  if [[ $default == "y" ]]; then hint="[Y/n]"; else hint="[y/N]"; fi
  while true; do
    prompt "$p $hint"
    read -r reply </dev/tty
    reply="${reply:-$default}"
    case "$reply" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      [Nn]|[Nn][Oo])     return 1 ;;
      *) warn "Please answer y or n." ;;
    esac
  done
}

ask_menu() {  # ask_menu "Prompt" default_index "label1" "label2" ... → prints selected label on stdout
  local p="$1" def="$2"; shift 2
  local items=("$@") n="${#@}" i reply
  printf '\n' >&2
  for i in "${!items[@]}"; do
    local marker="  "
    (( i+1 == def )) && marker="${GREEN}* ${RESET}"
    printf '  %s  %s%d)%s %s\n' "$marker" "$BOLD$CYAN" "$((i+1))" "$RESET" "${items[$i]}" >&2
  done
  while true; do
    prompt "$p (1-$n, default=$def)"
    read -r reply </dev/tty
    reply="${reply:-$def}"
    if [[ "$reply" =~ ^[0-9]+$ ]] && (( reply >= 1 && reply <= n )); then
      printf '%s\n' "${items[$((reply-1))]}"
      return 0
    fi
    warn "Enter a number between 1 and $n (or press Enter for default)."
  done
}

ask_hidden() {  # ask_hidden "Prompt" → prints secret on stdout
  local p="$1" reply
  prompt "$p"
  stty -echo 2>/dev/null
  read -r reply </dev/tty
  stty echo 2>/dev/null
  printf '\n' >&2
  printf '%s' "$reply"
}

# ---- Platform detection ---------------------------------------------------
banner "pid-gated-agent-learning — interactive setup"

step "Detecting platform"
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
case "$OS_NAME" in
  Darwin)                       PLATFORM="macOS" ;;
  Linux)
    if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
      PLATFORM="Linux (WSL)"
    else
      PLATFORM="Linux"
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*)         PLATFORM="Windows (Git Bash/MSYS)" ;;
  FreeBSD|NetBSD|OpenBSD)       PLATFORM="BSD" ;;
  *)                            PLATFORM="$OS_NAME" ;;
esac
ok "$PLATFORM ($(uname -m 2>/dev/null || echo unknown))"

if [[ "$PLATFORM" == "Windows"* ]]; then
  warn "Native Windows is not fully supported by this script."
  info "Recommended on Windows: use WSL2 and re-run this script, or follow the"
  info "manual steps in README.md (install uv, run 'uv sync', copy .env.template)."
  ask_yn "Attempt to continue anyway?" n || exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
info "repo: $REPO_ROOT"

# ---- uv ------------------------------------------------------------------
step "Checking uv"
if command -v uv >/dev/null 2>&1; then
  ok "uv found: $(uv --version)"
else
  warn "uv is not installed."
  info "uv is Astral's fast Python package manager (https://docs.astral.sh/uv/)."

  # Build a list of install methods that are actually available on this system.
  METHODS=()
  command -v brew >/dev/null 2>&1 && METHODS+=("brew install uv")
  command -v curl >/dev/null 2>&1 && METHODS+=("curl | sh (official installer)")
  command -v wget >/dev/null 2>&1 && METHODS+=("wget | sh (official installer)")
  command -v pipx >/dev/null 2>&1 && METHODS+=("pipx install uv")
  command -v pip  >/dev/null 2>&1 && METHODS+=("pip install --user uv")
  METHODS+=("skip (I'll install uv myself and re-run)")

  if ! ask_yn "Install uv now?" y; then
    err "uv is required. Install from https://docs.astral.sh/uv/ and re-run."
    exit 1
  fi

  CHOICE="$(ask_menu "Which install method?" 1 "${METHODS[@]}")"
  case "$CHOICE" in
    "brew install uv")
      brew install uv || { err "brew install failed."; exit 1; } ;;
    "curl | sh"*)
      curl -LsSf https://astral.sh/uv/install.sh | sh || { err "curl install failed."; exit 1; } ;;
    "wget | sh"*)
      wget -qO- https://astral.sh/uv/install.sh | sh || { err "wget install failed."; exit 1; } ;;
    "pipx install uv")
      pipx install uv || { err "pipx install failed."; exit 1; } ;;
    "pip install --user uv")
      pip install --user uv || { err "pip install failed."; exit 1; } ;;
    skip*)
      info "Install uv manually from https://docs.astral.sh/uv/getting-started/installation/"
      info "then re-run this script."
      exit 0 ;;
  esac

  # Make freshly installed uv findable in the current shell.
  # shellcheck disable=SC1090
  [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  if command -v uv >/dev/null 2>&1; then
    ok "uv installed: $(uv --version)"
  else
    err "uv was installed but isn't on PATH. Open a new shell and re-run this script."
    exit 1
  fi
fi

# ---- Python version ------------------------------------------------------
step "Pick a Python version"
PY_CHOICE="$(ask_menu "Which Python?" 1 \
  "3.13 (latest supported)" \
  "3.12" \
  "3.11" \
  "3.10 (minimum)" \
  "auto (let uv decide)")"
case "$PY_CHOICE" in
  3.13*) PY_VERSION="3.13" ;;
  3.12*) PY_VERSION="3.12" ;;
  3.11*) PY_VERSION="3.11" ;;
  3.10*) PY_VERSION="3.10" ;;
  *)     PY_VERSION="" ;;
esac
if [[ -n "$PY_VERSION" ]]; then
  info "ensuring Python $PY_VERSION is available via uv…"
  uv python install "$PY_VERSION" >/dev/null 2>&1 || true
  ok "will use Python $PY_VERSION"
else
  ok "will let uv pick a Python"
fi

# ---- Providers + .env ----------------------------------------------------
step "Configure LLM providers"

if [[ -f .env ]]; then
  warn ".env already exists in the repo root."
  if ! ask_yn "Overwrite it with answers from this wizard?" n; then
    info "keeping existing .env; skipping provider configuration."
    SKIP_ENV=1
  fi
fi

if [[ -z "${SKIP_ENV:-}" ]]; then
  info "pick one or more providers to configure (keys are hidden while you type)"
  info "the providers themselves pick sensible default models — no need to specify one here"
  OPENAI_KEY=""
  ANTHROPIC_KEY=""
  GEMINI_KEY=""
  DEEPSEEK_KEY=""
  OLLAMA_ENDPOINT=""

  if ask_yn "Configure OpenAI?" y; then
    OPENAI_KEY="$(ask_hidden "  OPENAI_API_KEY (hidden):")"
    ok "OpenAI configured"
  fi
  if ask_yn "Configure Anthropic?" n; then
    ANTHROPIC_KEY="$(ask_hidden "  ANTHROPIC_API_KEY (hidden):")"
    ok "Anthropic configured"
  fi
  if ask_yn "Configure Google Gemini?" n; then
    GEMINI_KEY="$(ask_hidden "  GEMINI_API_KEY (hidden):")"
    ok "Gemini configured"
  fi
  if ask_yn "Configure DeepSeek?" n; then
    DEEPSEEK_KEY="$(ask_hidden "  DEEPSEEK_API_KEY (hidden):")"
    ok "DeepSeek configured"
  fi
  if ask_yn "Configure a local Ollama endpoint?" n; then
    prompt "  OLLAMA_ENDPOINT (blank = http://localhost:11434):"
    read -r OLLAMA_ENDPOINT </dev/tty
    OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
    ok "Ollama endpoint: $OLLAMA_ENDPOINT"
  fi

  if [[ -z "$OPENAI_KEY$ANTHROPIC_KEY$GEMINI_KEY$DEEPSEEK_KEY$OLLAMA_ENDPOINT" ]]; then
    warn "No providers were configured. .env will not be written."
    warn "The notebook will fail on its first provider call until you re-run this script."
  else
    info "writing .env (gitignored)…"
    {
      echo "# Generated by scripts/setup.sh on $(date). Keys are secrets — do NOT commit."
      echo ""
      [[ -n "$ANTHROPIC_KEY"   ]] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_KEY"
      [[ -n "$GEMINI_KEY"      ]] && echo "GEMINI_API_KEY=$GEMINI_KEY"
      [[ -n "$DEEPSEEK_KEY"    ]] && echo "DEEPSEEK_API_KEY=$DEEPSEEK_KEY"
      [[ -n "$OPENAI_KEY"      ]] && echo "OPENAI_API_KEY=$OPENAI_KEY"
      [[ -n "$OLLAMA_ENDPOINT" ]] && echo "OLLAMA_ENDPOINT=$OLLAMA_ENDPOINT"
    } > .env
    chmod 600 .env
    ok ".env written (chmod 600, $(grep -cE '^[A-Z_]+=' .env) vars)"
  fi
fi

# ---- uv sync -------------------------------------------------------------
step "Installing dependencies with uv sync"
SYNC_ARGS=()
[[ -n "$PY_VERSION" ]] && SYNC_ARGS+=(--python "$PY_VERSION")
if uv sync "${SYNC_ARGS[@]}"; then
  ok "dependencies installed"
else
  err "uv sync failed."
  exit 1
fi

# ---- Smoke test ----------------------------------------------------------
step "Running smoke tests"
if uv run "${SYNC_ARGS[@]}" pytest -q >/tmp/pid_setup_pytest.log 2>&1; then
  ok "pytest green ($(tail -1 /tmp/pid_setup_pytest.log))"
else
  warn "pytest reported failures — see /tmp/pid_setup_pytest.log"
fi

# ---- Launch notebook -----------------------------------------------------
step "Launch the quickstart notebook?"
if ask_yn "Open examples/quickstart.ipynb in Jupyter now?" y; then
  # Export .env into our shell so the Jupyter server (and the kernel it spawns)
  # inherit the provider credentials — belt-and-braces alongside the notebook's
  # own load_dotenv() in cell 4.
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    ok "exported $(grep -cE '^[A-Z_]+=' .env) env vars from .env into the Jupyter server environment"
  else
    warn ".env not found; the notebook will fail at the first provider call unless keys are set another way."
  fi
  info "launching: uv run jupyter notebook examples/quickstart.ipynb"
  info "(Ctrl-C in this terminal to stop the Jupyter server)"
  exec uv run "${SYNC_ARGS[@]}" jupyter notebook examples/quickstart.ipynb
else
  banner "You're all set"
  info "Activate the environment with:    source .venv/bin/activate"
  info "Load secrets into your shell:     set -a; source .env; set +a"
  info "Launch the notebook later with:   uv run jupyter notebook examples/quickstart.ipynb"
  info "Re-run cross-version checks with: ./scripts/platform_compatibility_check.sh"
fi
