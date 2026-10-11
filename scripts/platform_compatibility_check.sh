#!/usr/bin/env bash
# Developer-side compatibility check: runs ruff, mypy, pytest, build, and a
# no-extras import smoke test against every supported Python version using uv.
#
# Mirrors what .github/workflows/ci.yml runs in CI, so you can catch Python-
# version regressions locally before pushing.
#
# Usage:
#   ./scripts/platform_compatibility_check.sh                   # test all supported versions
#   ./scripts/platform_compatibility_check.sh 3.11 3.12         # test only these versions
#
# Requires: uv (https://docs.astral.sh/uv/).

set -u

SUPPORTED_VERSIONS=("3.10" "3.11" "3.12" "3.13")
VERSIONS=("${@:-${SUPPORTED_VERSIONS[@]}}")

if [[ $# -gt 0 ]]; then
  VERSIONS=("$@")
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Platform:      $(uname -s) $(uname -m)"
echo "uv version:    $(uv --version)"
echo "Repo root:     $REPO_ROOT"
echo "Testing:       ${VERSIONS[*]}"
echo

PASS=()
FAIL=()

run_step() {
  local label="$1"
  shift
  printf "  %-28s" "$label"
  if output=$("$@" 2>&1); then
    echo "ok"
    return 0
  else
    echo "FAIL"
    echo "$output" | sed 's/^/      /'
    return 1
  fi
}

for v in "${VERSIONS[@]}"; do
  echo "========================================"
  echo "Python $v"
  echo "========================================"
  version_ok=1

  run_step "ruff check"       uv run --python "$v" --quiet ruff check .            || version_ok=0
  run_step "ruff format"      uv run --python "$v" --quiet ruff format --check .   || version_ok=0
  run_step "mypy src"         uv run --python "$v" --quiet mypy src                || version_ok=0
  run_step "pytest"           uv run --python "$v" --quiet pytest -q               || version_ok=0

  # No-extras install smoke test in a throwaway venv.
  TMP_VENV="$(mktemp -d)/venv"
  if uv venv --python "$v" "$TMP_VENV" >/dev/null 2>&1 \
     && VIRTUAL_ENV="$TMP_VENV" uv pip install --quiet . >/dev/null 2>&1; then
    if "$TMP_VENV/bin/python" - <<'PY' >/dev/null 2>&1
import pid_gated_agent_learning as p
assert p.__version__
# Core objects must exist with zero deps.
for name in ("IntegralGate", "ProportionalGate", "DerivativeGate",
            "Scheduler", "Parity", "GuessNumber", "MovingTarget",
            "MockProvider", "CachedProvider"):
    assert hasattr(p, name), name
# Each provider adapter raises a helpful ImportError when its SDK is absent.
for cls, needle in (
    (p.OpenAIProvider, "openai"),
    (p.AnthropicProvider, "anthropic"),
    (p.GeminiProvider, "google-genai"),
    (p.DeepSeekProvider, "openai"),
    (p.OllamaProvider, "httpx"),
):
    try:
        cls(api_key="dummy") if cls is not p.OllamaProvider else cls()
    except ImportError as e:
        assert needle in str(e), str(e)
    else:
        raise AssertionError(f"{cls.__name__} should have raised ImportError")
# MockProvider must work without any deps.
assert p.MockProvider(response="hi").generate("x") == "hi"
PY
    then
      printf "  %-28s%s\n" "no-extras install" "ok"
    else
      printf "  %-28s%s\n" "no-extras install" "FAIL"
      version_ok=0
    fi
  else
    printf "  %-28s%s\n" "no-extras install" "FAIL (venv/install)"
    version_ok=0
  fi
  rm -rf "$(dirname "$TMP_VENV")"

  # Build sdist + wheel (once per version is overkill, but it catches build.py
  # regressions that depend on the Python version).
  run_step "uv build"         uv build --python "$v" --quiet                       || version_ok=0

  if [[ $version_ok -eq 1 ]]; then
    PASS+=("$v")
  else
    FAIL+=("$v")
  fi
  echo
done

echo "========================================"
echo "Summary"
echo "========================================"
echo "Passed: ${PASS[*]:-(none)}"
echo "Failed: ${FAIL[*]:-(none)}"

if [[ ${#FAIL[@]} -gt 0 ]]; then
  exit 1
fi
