#!/bin/bash
# AutoCut installer (macOS). Run with:  bash install.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$HOME/.autocut"
VENV="$APP_DIR/venv"
BIN_DIR="$HOME/.local/bin"
MODEL="${AUTOCUT_MODEL:-small}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---- Python 3.9+
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3 /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.9+ not found. Run:  xcode-select --install   (or: brew install python) and try again."
  exit 1
fi

say "1/3  Python environment → $VENV  (using $("$PY" --version 2>&1))"
mkdir -p "$APP_DIR"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet "faster-whisper>=1.1,<2" "imageio-ffmpeg>=0.5" numpy tqdm
"$VENV/bin/python" -c "import faster_whisper, imageio_ffmpeg, numpy; print('   dependencies OK')"
cp "$HERE/autocut.py" "$APP_DIR/autocut.py"

say "2/3  'autocut' command → $BIN_DIR/autocut"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/autocut" <<EOF
#!/bin/bash
exec "$VENV/bin/python" "$APP_DIR/autocut.py" "\$@"
EOF
chmod +x "$BIN_DIR/autocut"
NEW_SHELL=0
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    case "${SHELL:-}" in */bash) RC="$HOME/.bash_profile" ;; *) RC="$HOME/.zprofile" ;; esac
    if ! grep -qs '.local/bin' "$RC"; then
      printf '\n# added by AutoCut installer\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$RC"
      echo "   added ~/.local/bin to PATH in $RC"
    fi
    NEW_SHELL=1
    ;;
esac

say "3/3  Downloading the Whisper '$MODEL' model (one time, a few hundred MB)"
if ! "$VENV/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('$MODEL', device='cpu', compute_type='int8')"; then
  echo "   download failed. It will be retried automatically on the first 'autocut' run."
fi

say "Done."
if [ "$NEW_SHELL" = 1 ]; then
  echo "Open a NEW Terminal window, then:"
else
  echo "Then:"
fi
echo "  autocut /path/to/footage.mp4"
