# Setup Guide for Real-Time AI Meeting Assistant

## Hardware Requirements
- Lenovo ThinkPad P16 Gen 2 or similar with a microphone.
- GPU (e.g., NVIDIA RTX 5000) enhances performance.

## Software Requirements
- Ubuntu 24.04.3 LTS.
- Python 3.10+.
- Dependencies in `requirements.txt`.

## Installation Steps
1. Update: `sudo apt update && sudo apt upgrade`
2. Install tools: `sudo apt install python3-dev python3-pip portaudio19-dev`
3. Follow `README.md` installation.
4. Configure audio: Check groups, test with `arecord`.
5. API keys:
   - xAI API key: From https://console.x.ai
   - Hugging Face token: From https://huggingface.co/settings/tokens (for speaker diarization)
6. Create `.env` file with your API keys.

## Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API Configuration
Create a `.env` file in the project root:
```bash
# xAI Grok API Key (required)
XAI_API_KEY=your-xai-api-key-here

# Hugging Face Token for speaker diarization (optional)
HF_TOKEN=your-huggingface-token-here
```

## Speaker Diarization Setup (Optional)
1. Get a Hugging Face token from https://huggingface.co/settings/tokens
2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the repository terms
3. Add the token to your `.env` file as `HF_TOKEN`

## Troubleshooting
- ALSA errors: `pkill -9 arecord; pulseaudio -k && pulseaudio --start`
- PyAudio: `pip install pyaudio==0.2.14`
- GPU: `nvidia-smi; python -c "import torch; print(torch.cuda.is_available())"`
- Speaker diarization issues: Check HF_TOKEN validity and repository access

## Known Issues
- Device index may need adjustment.
- LLM format depends on API.
- Speaker diarization may fail if HF_TOKEN is invalid or terms not accepted.

## Contact
- Via GitHub Issues.
