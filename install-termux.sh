#!/data/data/com.termux/files/usr/bin/bash
#
# Nemotron Farsi ASR API - Termux installer
# Repository: https://github.com/mer30hamid/nemotron-asr-streaming-farsi-api
#
# Install:
#   curl -fsSL https://raw.githubusercontent.com/mer30hamid/nemotron-asr-streaming-farsi-api/main/install-termux.sh | bash
#
# Run:
#   nemotron-farsi
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

MODEL_FILES=(
  "encoder.int8.onnx"
  "decoder.int8.onnx"
  "joiner.int8.onnx"
  "tokens.txt"
)

# Conservative minimum sizes, used to catch truncated downloads.
# These are not cryptographic checksums.
declare -A MIN_SIZE=(
  ["encoder.int8.onnx"]=600000000
  ["decoder.int8.onnx"]=6000000
  ["joiner.int8.onnx"]=1500000
  ["tokens.txt"]=10000
)

info()    { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
success() { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()    { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error()   { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; }

confirm() {
  local answer
  read -r -p "$1 [y/N]: " answer
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
}
trap 'on_error $LINENO' ERR

# This installer is specifically for Termux.
if [[ -z "${PREFIX:-}" || ! -d "${PREFIX:-/nonexistent}/bin" ]]; then
  die "Termux environment not detected. Run this script inside Termux."
fi

if [[ ! "$PREFIX" == "/data/data/com.termux/files/usr" &&
      ! "$PREFIX" == "/data/user/0/com.termux/files/usr" ]]; then
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

# Check available storage before attempting a large model download.
mkdir -p "$APP_DIR" "$MODEL_DIR"
AVAILABLE_KB="$(df -Pk "$APP_DIR" | awk 'NR==2 {print $4}')"
if [[ -n "$AVAILABLE_KB" ]] && (( AVAILABLE_KB < 750000 )); then
  warn "Less than approximately 750 MB is available on this filesystem."
  warn "The model download needs about 640 MB, plus temporary/free space."
  confirm "Continue anyway?" || die "Installation cancelled."
fi

# Install native Termux tools.
info "Updating Termux package lists..."
pkg update -y

info "Installing Python, Git, curl and FFmpeg..."
pkg install -y python git curl ffmpeg

command -v python >/dev/null 2>&1 || die "Python was not installed."
command -v git >/dev/null 2>&1 || die "Git was not installed."
command -v curl >/dev/null 2>&1 || die "curl was not installed."
command -v ffmpeg >/dev/null 2>&1 || die "FFmpeg was not installed."

PYTHON_VERSION="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
info "Termux Python version: $PYTHON_VERSION"

python - <<'PY' || die "Python 3.11 or newer is required by this project."
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY

# Fetch or update the application source.
if [[ -d "$APP_DIR/.git" ]]; then
  info "Updating existing Git checkout..."
  git -C "$APP_DIR" pull --ff-only
elif [[ -e "$APP_DIR" ]] && [[ -n "$(find "$APP_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "$APP_DIR already exists but is not a Git checkout. Move it aside and retry."
else
  info "Cloning the API repository..."
  rmdir "$APP_DIR" 2>/dev/null || true
  git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi

API_SCRIPT="$APP_DIR/nemotron_asr_streaming_farsi_api.py"
REQUIREMENTS="$APP_DIR/requirements.txt"

[[ -f "$API_SCRIPT" ]] || die "API Python file was not found in the repository."
[[ -f "$REQUIREMENTS" ]] || die "requirements.txt was not found in the repository."

# Validate model files by minimum expected size.
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
    size="$(du -h "$MODEL_DIR/$name" | awk '{print $1}')"
    success "$name already present ($size)."
  else
    if [[ -e "$MODEL_DIR/$name" ]]; then
      warn "$name exists but is smaller than expected; it may be incomplete."
    else
      warn "$name is missing."
    fi
    MISSING_FILES+=("$name")
  fi
done

# Ask once before downloading any missing/invalid model files.
if (( ${#MISSING_FILES[@]} > 0 )); then
  echo
  warn "The model is incomplete. The encoder file alone is about 628 MB."
  echo "Files that need to be downloaded or repaired:"
  for name in "${MISSING_FILES[@]}"; do
    echo "  - $name"
  done
  echo

  confirm "Download the missing/incomplete model files now?" ||
    die "Model download was declined. No server installation was completed."

  for name in "${MISSING_FILES[@]}"; do
    target="$MODEL_DIR/$name"
    url="$HF_BASE_URL/$name"

    info "Downloading $name (an existing partial download will be resumed)..."

    # --continue-at resumes a partial file when the server supports HTTP ranges.
    # A failed transfer leaves the partial file in place for the next run.
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

    model_file_is_valid "$name" ||
      die "$name failed the size check after download. Re-run the installer to retry."

    success "$name downloaded and passed the size check."
  done
else
  success "All model files are already present."
fi

# Final model validation.
for name in "${MODEL_FILES[@]}"; do
  model_file_is_valid "$name" || die "Model verification failed: $name"
done
success "All four model files are present."

# Create/refresh the Python virtual environment.
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  info "Creating Python virtual environment..."
  python -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
[[ -x "$VENV_PYTHON" ]] || die "Could not create the Python virtual environment."

info "Upgrading pip, setuptools and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

info "Installing project Python requirements..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"

# Verify the exact runtime imports used by the API.
info "Checking NumPy and sherpa-onnx imports..."
"$VENV_PYTHON" - <<'PY'
import numpy
import sherpa_onnx
print("NumPy:", numpy.__version__)
print("sherpa-onnx:", getattr(sherpa_onnx, "__version__", "installed"))
PY

# Create config only on first installation; preserve user edits on upgrades.
mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CONFIG_FILE" ]]; then
  cat > "$CONFIG_FILE" <<EOF
# Nemotron Farsi ASR settings
# The server binds to localhost by default. Keep it that way unless needed.
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

# Ensure MODEL_PATH points to the model folder even if the user config is
# edited later. The launcher sources the config file and then starts Python.
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

# The API script reads MODEL_PATH (not MODEL_DIR).
export MODEL_PATH="${MODEL_PATH:-$APP_DIR/models/nemotron-asr-streaming-farsi-onnx}"

exec "$VENV_PYTHON" "$API_SCRIPT"
LAUNCHER_EOF

chmod 755 "$LAUNCHER"
hash -r 2>/dev/null || true

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
echo "  http://127.0.0.1:8765/v1/audio/transcriptions"
echo
echo "Configuration:"
echo "  nano $CONFIG_FILE"
echo
echo "Stop the server with Ctrl+C."
echo
warn "Keep Termux open while the API server is running."
