#!/data/data/com.termux/files/usr/bin/bash
#
# Nemotron Farsi ASR API - Termux installer
# Repository: https://github.com/mer30hamid/nemotron-asr-streaming-farsi-api
#
# Install:
#   curl -fsSL https://raw.githubusercontent.com/mer30hamid/nemotron-asr-streaming-farsi-api/main/install-termux.sh | bash
#
# Run after install:
#   nemotron-farsi
#
# This installer is native Termux only. Docker is not used.
#

set -Eeuo pipefail

REPO_URL="https://github.com/mer30hamid/nemotron-asr-streaming-farsi-api.git"
HF_BASE_URL="https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx/resolve/main"

APP_DIR="$HOME/nemotron-farsi"
MODEL_DIR="$APP_DIR/models/nemotron-asr-streaming-farsi-onnx"
VENV_DIR="$APP_DIR/venv"
CONFIG_DIR="$HOME/.config/nemotron-farsi"
CONFIG_FILE="$CONFIG_DIR/env"
LAUNCHER="$PREFIX/bin/nemotron-farsi"

# Exact files required by nemotron_asr_streaming_farsi_api.py
MODEL_FILES=(
  "encoder.int8.onnx"
  "decoder.int8.onnx"
  "joiner.int8.onnx"
  "tokens.txt"
)

# Approximate real sizes (from Hugging Face x-linked-size headers, Nov 2026).
# Used only as a conservative lower bound to reject truncated downloads.
# These are NOT cryptographic checksums; the project does not publish SHA256.
declare -A MIN_SIZE=(
  ["encoder.int8.onnx"]=640000000   # real ~657601519
  ["decoder.int8.onnx"]=7000000     # real ~7257753
  ["joiner.int8.onnx"]=1600000      # real ~1735861
  ["tokens.txt"]=15000              # real ~16062
)

info()    { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
success() { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()    { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error()   { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; }

confirm() {
  local answer
  # Must read from the real terminal. When the installer is run via
  #   curl ... | bash
  # stdin is the pipe, so a plain `read` would always get EOF and treat
  # the answer as "no". /dev/tty is the controlling terminal.
  if [[ -r /dev/tty ]]; then
    read -r -p "$1 [y/N]: " answer < /dev/tty
  else
    # Fallback (non-interactive / no tty): treat as declined.
    printf '%s [y/N]: \n' "$1" >&2
    answer=""
  fi
  [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]
}

die() {
  error "$*"
  exit 1
}

on_error() {
  local line="$1"
  error "Installation stopped at line $line."
  error "Fix the reported problem and run the installer again."
  error "Existing model files were not deleted."
}
trap 'on_error $LINENO' ERR

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

if [[ -z "${PREFIX:-}" || ! -d "${PREFIX:-/nonexistent}/bin" ]]; then
  die "Termux environment not detected. Run this script inside Termux."
fi

if [[ "$PREFIX" != "/data/data/com.termux/files/usr" &&
      "$PREFIX" != "/data/user/0/com.termux/files/usr" ]]; then
  warn "Unexpected Termux PREFIX: $PREFIX"
fi

clear 2>/dev/null || true
echo "=================================================="
echo "       Nemotron Farsi ASR - Termux Installer"
echo "=================================================="
echo
echo "Install directory : $APP_DIR"
echo "Model directory   : $MODEL_DIR"
echo

# Ensure base directories exist early (safe for pre-copied models).
mkdir -p "$APP_DIR" "$MODEL_DIR"

# Rough free-space check before large downloads.
AVAILABLE_KB="$(df -Pk "$APP_DIR" 2>/dev/null | awk 'NR==2 {print $4}')" || AVAILABLE_KB=""
if [[ -n "$AVAILABLE_KB" ]] && (( AVAILABLE_KB < 800000 )); then
  warn "Less than ~800 MB free on this filesystem."
  warn "The encoder alone is ~628 MB; extra space is recommended."
  confirm "Continue anyway?" || die "Installation cancelled by user."
fi

# ---------------------------------------------------------------------------
# System packages (native Termux)
# ---------------------------------------------------------------------------

info "Updating Termux package lists..."
pkg update -y
pkg upgrade -y

info "Installing Python, NumPy, Git, curl and FFmpeg..."
pkg install -y python python-numpy git curl ffmpeg libexpat openssl

command -v python >/dev/null 2>&1 || die "Python was not installed."
command -v git    >/dev/null 2>&1 || die "Git was not installed."
command -v curl   >/dev/null 2>&1 || die "curl was not installed."
command -v ffmpeg >/dev/null 2>&1 || die "FFmpeg was not installed."

PYTHON_VERSION="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
info "Termux Python version: $PYTHON_VERSION"

python - <<'PY' || die "Python 3.11 or newer is required."
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY

# NumPy must come from Termux, not from pip. PyPI wheels for NumPy are
# built for glibc Linux, so on Termux (Android/Bionic) pip falls back to
# building NumPy from source, and that build fails.
python - <<'PY' || die "NumPy from Termux (python-numpy) was not installed."
import numpy
PY
NUMPY_VERSION="$(python -c 'import numpy; print(numpy.__version__)')"
info "Termux NumPy version: $NUMPY_VERSION"

# ---------------------------------------------------------------------------
# Fetch / merge application source (never destroy existing models)
# ---------------------------------------------------------------------------

if [[ -d "$APP_DIR/.git" ]]; then
  info "Existing Git repository found. Updating with fast-forward only..."
  if ! git -C "$APP_DIR" pull --ff-only; then
    warn "git pull --ff-only failed (local changes or divergent history)."
    warn "Repository files left as-is. Continuing with current tree."
  fi
else
  # APP_DIR may already contain models that the user copied manually.
  # Clone into a temporary directory, then merge carefully.
  TMP_CLONE="${APP_DIR}.source-tmp.$$"

  info "Cloning repository into temporary directory..."
  git clone --depth 1 "$REPO_URL" "$TMP_CLONE"

  info "Merging repository files into $APP_DIR (preserving models)..."
  # cp -a copies the tree; existing model files under models/ are left alone
  # because they are not present in the repository.
  cp -a "$TMP_CLONE/." "$APP_DIR/"

  rm -rf "$TMP_CLONE"
  success "Repository files merged. Existing model files were preserved."
fi

API_SCRIPT="$APP_DIR/nemotron_asr_streaming_farsi_api.py"
REQUIREMENTS="$APP_DIR/requirements.txt"

[[ -f "$API_SCRIPT" ]]   || die "API Python file not found: $API_SCRIPT"
[[ -f "$REQUIREMENTS" ]] || die "requirements.txt not found: $REQUIREMENTS"

# ---------------------------------------------------------------------------
# Model file validation and optional download
# ---------------------------------------------------------------------------

model_file_is_valid() {
  local name="$1"
  local path="$MODEL_DIR/$name"
  local size
  [[ -f "$path" ]] || return 1
  size="$(wc -c < "$path")"
  (( size >= MIN_SIZE[$name] ))
}

MISSING_FILES=()
echo
info "Checking the four required model files..."

for name in "${MODEL_FILES[@]}"; do
  if model_file_is_valid "$name"; then
    size_h="$(du -h "$MODEL_DIR/$name" | awk '{print $1}')"
    success "$name already present ($size_h)."
  else
    if [[ -e "$MODEL_DIR/$name" ]]; then
      warn "$name exists but is smaller than expected; it may be incomplete."
    else
      warn "$name is missing."
    fi
    MISSING_FILES+=("$name")
  fi
done

if (( ${#MISSING_FILES[@]} > 0 )); then
  echo
  echo "Model files are missing."
  echo
  warn "The following files need to be downloaded or repaired:"
  for name in "${MISSING_FILES[@]}"; do
    echo "  - $name"
  done
  echo
  echo "Approximate total download size: ~640 MB"
  echo "Source: https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx"
  echo
  echo "Partial downloads will be resumed with HTTP Range (curl --continue-at)."
  echo

  # Exact prompt required by the installation specification.
  if ! confirm "Do you want to download them now?"; then
    echo
    warn "Model download declined by user."
    warn "Existing files were not modified or deleted."
    warn "Re-run the installer after placing the model files, or answer yes next time."
    exit 0
  fi

  mkdir -p "$MODEL_DIR"

  for name in "${MISSING_FILES[@]}"; do
    target="$MODEL_DIR/$name"
    url="$HF_BASE_URL/$name"

    info "Downloading $name (resume supported)..."

    # --continue-at - resumes a partial file when the server supports ranges.
    # A failed transfer leaves the partial file for the next run.
    curl \
      --fail \
      --location \
      --show-error \
      --retry 5 \
      --retry-delay 3 \
      --connect-timeout 30 \
      --continue-at - \
      --output "$target" \
      "$url"

    if ! model_file_is_valid "$name"; then
      die "$name failed the size check after download. Re-run the installer to retry."
    fi
    success "$name downloaded and passed the size check."
  done
else
  success "All model files are already present."
fi

# Final verification
for name in "${MODEL_FILES[@]}"; do
  model_file_is_valid "$name" || die "Model verification failed: $name"
done
success "All four model files are present and passed the size check."
echo
info "Note: only minimum-size checks are performed. The project does not publish SHA256 checksums."

# ---------------------------------------------------------------------------
# Python virtual environment and dependencies
# ---------------------------------------------------------------------------
#
# Termux Python often ships without a working ensurepip. Creating a venv the
# normal way then fails with:
#   .../python -m ensurepip --upgrade --default-pip
#   returned non-zero exit status 1
# The reliable approach on Termux is:
#   1. python -m venv --without-pip --system-site-packages <dir>
#   2. bootstrap pip with get-pip.py into that venv
#
# --system-site-packages is required here. An isolated venv cannot see
# the Termux NumPy package, so pip would try to install NumPy from PyPI
# and fall back to building it from source, which fails on Termux.
#
# A venv created by an older installer, or by the failed run, is
# isolated. Recreate it so the new run actually gets a venv that can
# see the Termux NumPy package.

RECREATE_VENV=0
if [[ -x "$VENV_DIR/bin/python" ]]; then
  if ! grep -qs '^include-system-site-packages = true$' "$VENV_DIR/pyvenv.cfg"; then
    RECREATE_VENV=1
  fi
else
  RECREATE_VENV=1
fi

if (( RECREATE_VENV == 1 )); then
  info "Creating Python virtual environment (without ensurepip)..."
  # Remove a half-created venv, or an isolated venv, from a previous run.
  if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
  fi
  python -m venv --without-pip --system-site-packages "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
[[ -x "$VENV_PYTHON" ]] || die "Could not create the Python virtual environment."

# Fail here with a clear message if the venv cannot see Termux NumPy,
# instead of letting pip try to build NumPy from source later.
"$VENV_PYTHON" -c 'import numpy' 2>/dev/null || die "NumPy is not visible inside the virtual environment. Re-run the installer after 'pkg install python-numpy'."

# Bootstrap pip if the venv does not have it yet (common on Termux).
if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  info "Bootstrapping pip into the virtual environment..."
  GET_PIP_TMP="$(mktemp "${TMPDIR:-/tmp}/get-pip.XXXXXX.py")"
  curl -fsSL "https://bootstrap.pypa.io/get-pip.py" -o "$GET_PIP_TMP"
  "$VENV_PYTHON" "$GET_PIP_TMP"
  rm -f "$GET_PIP_TMP"
fi

"$VENV_PYTHON" -m pip --version >/dev/null 2>&1 ||
  die "pip is not available inside the virtual environment."

info "Upgrading pip, setuptools and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

info "Installing project Python requirements (sherpa-onnx; NumPy comes from Termux)..."
# Install everything from requirements.txt except NumPy. NumPy is
# already installed from Termux and visible in this venv, so pip must
# not install it, upgrade it, or build it from source. sherpa-onnx
# does not depend on NumPy, so removing that one line removes nothing
# else. requirements.txt itself is left unchanged: on Docker and on
# desktop Linux the pip NumPy wheel works fine, only Termux is special.
# --only-binary=:all: is a guard: if a package has no wheel for this
# Termux Python, pip fails fast with a clear error instead of starting
# a source build that cannot succeed here.
PIP_REQUIREMENTS_TMP="$(mktemp "${TMPDIR:-/tmp}/requirements-no-numpy.XXXXXX.txt")"
# Plain prefix match on purpose. The precise pattern with \(...\|...\)
# does not work with the grep in Termux ("Unmatched ( or \("), so keep
# this filter simple. requirements.txt here has only sherpa-onnx and
# numpy, so a line starting with numpy is the numpy line.
grep -vi '^numpy' "$REQUIREMENTS" > "$PIP_REQUIREMENTS_TMP" || true
# Never let a failed filter silently produce an empty file: pip would
# then install nothing and the failure would only show up later at
# the import check. Stop here instead, with a clear message.
[[ -s "$PIP_REQUIREMENTS_TMP" ]] || die "Filtered requirements file is empty. The NumPy filter step failed; nothing was installed."
"$VENV_PYTHON" -m pip install --only-binary=:all: -r "$PIP_REQUIREMENTS_TMP"
rm -f "$PIP_REQUIREMENTS_TMP"

info "Verifying imports used by the API..."
"$VENV_PYTHON" - <<'PY'
import numpy
import sherpa_onnx
print("NumPy      :", numpy.__version__)
print("sherpa-onnx:", getattr(sherpa_onnx, "__version__", "installed"))
PY

# ---------------------------------------------------------------------------
# Configuration (preserve user edits on re-runs)
# ---------------------------------------------------------------------------

mkdir -p "$CONFIG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
  cat > "$CONFIG_FILE" <<EOF
# Nemotron Farsi ASR settings for Termux
# Edit this file and restart the server to apply changes.
# The server binds to localhost by default.

export MODEL_PATH="$MODEL_DIR"
export ASR_HOST="127.0.0.1"
export ASR_PORT="8765"
export ASR_THREADS="4"
export ASR_SAMPLE_RATE="16000"
export ASR_LANGUAGE="fa-IR"
export ASR_DECODING_METHOD="greedy_search"
export ASR_MAX_UPLOAD_MB="50"
export LOG_LEVEL="INFO"
EOF
  success "Created default configuration: $CONFIG_FILE"
else
  info "Existing configuration preserved: $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Launcher script
# ---------------------------------------------------------------------------

cat > "$LAUNCHER" <<'LAUNCHER_EOF'
#!/data/data/com.termux/files/usr/bin/bash
set -Eeuo pipefail

APP_DIR="$HOME/nemotron-farsi"
VENV_PYTHON="$APP_DIR/venv/bin/python"
API_SCRIPT="$APP_DIR/nemotron_asr_streaming_farsi_api.py"
CONFIG_FILE="$HOME/.config/nemotron-farsi/env"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[ERROR] Python virtual environment not found. Re-run the installer." >&2
  exit 1
fi

if [[ ! -f "$API_SCRIPT" ]]; then
  echo "[ERROR] API script not found: $API_SCRIPT" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

# The API reads MODEL_PATH (not MODEL_DIR).
export MODEL_PATH="${MODEL_PATH:-$APP_DIR/models/nemotron-asr-streaming-farsi-onnx}"

exec "$VENV_PYTHON" "$API_SCRIPT"
LAUNCHER_EOF

chmod 755 "$LAUNCHER"
hash -r 2>/dev/null || true

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo
echo "=================================================="
success "Nemotron Farsi ASR installation completed."
echo "=================================================="
echo
echo "Start the server:"
echo "  nemotron-farsi"
echo
echo "Health check (while the server is running):"
echo "  curl http://127.0.0.1:8765/"
echo
echo "Transcription endpoint:"
echo "  POST http://127.0.0.1:8765/v1/audio/transcriptions"
echo
echo "Configuration file:"
echo "  $CONFIG_FILE"
echo
echo "Stop the server with Ctrl+C."
echo
warn "Keep Termux open (or use a wake-lock / tmux) while the API is running."
echo
