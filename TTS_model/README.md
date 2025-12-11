# TTS Model Quickstart

Simple notes on how to set up and run the local NeMo TTS script.

## Prerequisites
- Python 3.12 (matching the current `.venv`)
- `ffmpeg` for best audio tooling: `brew install ffmpeg`

## Create and activate the virtual environment
```bash
cd /Users/saidharan/TTS_model
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies
```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Download models manually (already done)
Files are stored in `models/`:
- `models/tts_en_fastpitch.nemo`
- `models/tts_en_hifigan.nemo`

If you need to re-download:
```bash
mkdir -p models
curl -L "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.8.1/files/tts_en_fastpitch_align.nemo" -o models/tts_en_fastpitch.nemo
curl -L "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_hifigan/versions/1.0.0rc1/files/tts_hifigan.nemo" -o models/tts_en_hifigan.nemo
```

## Run a synthesis
`tts_inference.py` was removed; use `TTSEngine` directly in Python:
```python
from tts_engine import TTSEngine

tts = TTSEngine()  # auto-picks cuda if available else cpu
tts.synthesize("Hello world", "output.wav")
print("Saved to output.wav")
```

Or run an ad-hoc one-liner:
```bash
.venv/bin/python - <<'PY'
from tts_engine import TTSEngine
tts = TTSEngine()
tts.synthesize("Quick test from CLI", "output.wav")
print("Saved to output.wav")
PY
```

## Notes
- Model cache is local (`models/`). No network download needed once the `.nemo` files are present.
- Output audio defaults to 22050 Hz WAV.

