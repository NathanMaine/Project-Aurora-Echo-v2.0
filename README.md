# Real-Time AI Meeting Assistant

## Overview
This is a FastAPI-based application with WebSocket support that transcribes live audio from your microphone, summarizes meeting content, and extracts action items using the xAI Grok API.

# 🌟 Project Aurora Echo 🌟

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)]()

> *Illuminate your meetings with AI-powered real-time transcription, summarization, and action extraction.*

## ✨ Overview

**Project Aurora Echo** is a next-generation, real-time AI assistant platform designed to revolutionize meeting productivity. Built on FastAPI with WebSocket support, it seamlessly integrates advanced speech recognition, multiple large language model providers, and comprehensive observability to deliver unparalleled real-time insights.

Whether you're in a boardroom or working remotely, Aurora Echo captures live audio, transcribes conversations with speaker diarization, generates intelligent summaries, and extracts actionable items—all in real-time. Powered by cutting-edge AI technologies and optimized for performance, it's the ultimate tool for modern collaboration.

## 🚀 Key Features

- **🎙️ Real-Time Audio Transcription**: Capture and transcribe live audio streams using OpenAI Whisper, with GPU acceleration support
- **👥 Speaker Diarization**: Automatically identify and label different speakers in conversations using pyannote.audio
- **🤖 Multi-Provider LLM Integration**: Seamlessly switch between leading AI models:
  - Anthropic Claude
  - Azure OpenAI
  - Google Gemini
  - OpenAI GPT
  - vLLM (local inference)
  - xAI Grok
- **⚡ Async Inference Orchestration**: High-performance worker queue system for scalable, concurrent processing
- **📊 Comprehensive Observability**: Prometheus metrics, Traefik reverse proxy, and detailed logging
- **🐳 Docker-Ready**: Complete containerization with GPU support for production deployments
- **🔄 WebSocket Communication**: Real-time bidirectional communication for instant updates
- **🎵 Text-to-Speech Feedback**: Audible summaries and notifications
- **📈 Action Item Extraction**: Intelligent parsing of tasks, assignees, and deadlines
- **🛡️ Robust Error Handling**: Fallback provider system ensures reliability

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   FastAPI App    │◄──►│  Orchestrator   │
│   (WebSocket)   │    │   (WebSocket)    │    │  (Async Queue)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   ASR Service   │    │   LLM Providers │
                       │  (Whisper)      │    │  (Multi-Model)  │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Audio Buffer    │    │  Observability  │
                       │                 │    │  (Prometheus)   │
                       └─────────────────┘    └─────────────────┘
```

### Core Components

- **FastAPI Application**: RESTful API with WebSocket endpoints
- **ASR Service**: Speech-to-text using OpenAI Whisper
- **LLM Providers**: Modular provider system with fallback support
- **Orchestrator**: Async job queue for inference tasks
- **Audio Buffer**: Efficient audio data management
- **Observability**: Prometheus metrics and monitoring
- **Reverse Proxy**: Traefik for load balancing and SSL termination

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI with Uvicorn
- **Language**: Python 3.10+
- **Async**: asyncio for concurrent processing
- **WebSockets**: Real-time bidirectional communication

### AI & ML
- **Speech Recognition**: OpenAI Whisper
- **Speaker Diarization**: pyannote.audio
- **LLM Providers**: Anthropic, Azure OpenAI, Gemini, OpenAI, vLLM, xAI
- **GPU Acceleration**: PyTorch with CUDA support

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Reverse Proxy**: Traefik v3.0
- **Monitoring**: Prometheus
- **Audio Processing**: PyAudio, torchaudio

### Development
- **Environment**: python-dotenv
- **Data Processing**: pandas, numpy
- **TTS**: pyttsx3
- **Testing**: pytest (future)

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/dentity007/Project-Aurora-Echo.git
   cd Project-Aurora-Echo
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env  # Edit with your API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   # or
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Web UI: http://localhost
   - API Docs: http://localhost/docs
   - Metrics: http://localhost/metrics

## 🎯 Usage

1. **Open your browser** to `http://localhost:8000`
2. **Grant microphone permissions** when prompted
3. **Click "Record & Analyze"** to start real-time transcription
4. **View live results**: transcription, summary, and action items
5. **Receive audio feedback** for summaries

### API Endpoints

- `GET /`: Web interface
- `WebSocket /ws`: Real-time audio processing
- `GET /metrics`: Prometheus metrics
- `GET /docs`: Interactive API documentation

### Configuration

Set the following environment variables:

```env
# Required
XAI_API_KEY=your-xai-api-key
HF_TOKEN=your-huggingface-token  # For speaker diarization

# Optional LLM Provider Keys
ANTHROPIC_API_KEY=your-anthropic-key
AZURE_OPENAI_API_KEY=your-azure-key
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key

# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # For GPU acceleration
```

## 🔧 Development

### Project Structure

```
Project Aurora Echo/
├── app.py                 # Main FastAPI application
├── services/              # Core services
│   ├── asr_service.py     # Speech recognition
│   ├── llm_service.py     # LLM integration
│   ├── orchestrator.py    # Async job orchestration
│   ├── audio_buffer.py    # Audio data management
│   └── providers/         # LLM provider implementations
├── integrations/          # Workflow integrations
├── observability.py       # Prometheus metrics
├── docker/                # Container configuration
├── docs/                  # Documentation
├── static/                # Web assets
└── requirements.txt       # Python dependencies
```

### Running Tests

```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

**All Rights Reserved**

This project and all its contents are proprietary. No part of this software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright holder.

## 🙏 Acknowledgments

- OpenAI for Whisper model
- Hugging Face for pyannote.audio
- All LLM providers for their APIs
- FastAPI community for the excellent framework

---

*Built with ❤️ for the future of AI-assisted collaboration*
