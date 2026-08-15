#!/bin/bash
# Run any of the WSL validation tools with TensorFlow's CUDA chatter silenced.
#
#   bash tools/run_in_wsl.sh validate  <audio files...>
#   bash tools/run_in_wsl.sh bisect    <audio file> [musicnn|yamnet]
#   bash tools/run_in_wsl.sh fit       <audio file>
#
# From Windows:
#   wsl -d Ubuntu -- bash /mnt/c/Users/<you>/source/repos/ai-bridge-listener/tools/run_in_wsl.sh validate "<file>"
#
# Setup this expects (done once):
#   sudo apt install -y python3-venv python3-pip
#   python3 -m venv ~/essentia-venv
#   ~/essentia-venv/bin/pip install essentia-tensorflow numpy soundfile soxr
set -u
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""

PY="$HOME/essentia-venv/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -x "$PY" ] || { echo "no venv at $PY — see the header for setup"; exit 1; }

case "${1:-}" in
    validate) SCRIPT="$HERE/validate_mel_wsl.py" ;;
    bisect)   SCRIPT="$HERE/bisect_mel_wsl.py" ;;
    fit)      SCRIPT="$HERE/fit_log_wsl.py" ;;
    diagnose) SCRIPT="$HERE/diagnose_mel_wsl.py" ;;
    ab)       SCRIPT="$HERE/ab_mel_test.py" ;;
    *) echo "usage: $0 {validate|bisect|fit|diagnose|ab} <args...>"; exit 2 ;;
esac
shift

"$PY" "$SCRIPT" "$@" 2>&1 | grep -v -E \
    "tensorflow/|NUMA|dlerror|Skipping registering|StreamExecutor|coreClock|pciBusID|Could not load|Cannot dlopen|MusicExtractorSVM"
