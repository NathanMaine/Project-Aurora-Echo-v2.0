# Real-Time AI Meeting Assistant

## Overview
This is a FastAPI-based application with WebSocket support that transcribes live audio from your microphone, summarizes meeting content, and extracts action items using the xAI Grok API.

## Description
Real-Time AI Meeting Assistant
The Real-Time AI Meeting Assistant is an innovative FastAPI-based application designed to transform meeting productivity by transcribing live audio, generating concise summaries, and extracting actionable items in real time. Powered by the advanced Whisper model for speech-to-text and the xAI Grok API for natural language processing, this tool leverages cutting-edge AI to streamline collaboration and task management. Ideal for professionals, teams, and remote workers, it offers a seamless experience with text-to-speech feedback and a dynamic web interface.
Key Features

Live Audio Transcription: Captures and transcribes 5-second audio clips from your microphone using the Whisper model, optimized for GPU acceleration on compatible hardware.
Intelligent Summarization: Utilizes the xAI Grok API to distill meeting discussions into concise summaries, enhancing comprehension and retention.
Action Item Extraction: Automatically identifies and organizes tasks, assignees, and due dates, presented in an interactive table for easy tracking.
Text-to-Speech Feedback: Provides audible summaries via pyttsx3, ensuring accessibility and hands-free operation.
User-Friendly Interface: Built with FastAPI and WebSockets, offering a responsive web-based UI accessible at http://localhost:8000.

Technical Highlights
Developed on Ubuntu 24.04.3 LTS, the application harnesses a robust tech stack including PyAudio for audio capture, pandas for data management, and torch for GPU support. It’s designed to run efficiently on high-performance systems like the Lenovo ThinkPad P16 Gen 2, utilizing its NVIDIA RTX 5000 for accelerated processing. The code is modular, with cached resources to optimize performance, and includes comprehensive error handling for a reliable user experience.
Getting Started
Clone the repository, set up your environment with the provided requirements.txt, and configure your xAI API key to begin. The app is ready for local deployment, with detailed setup instructions in docs/Setup Guide for Real Time.md. Whether for personal use or team collaboration, this assistant adapts to your meeting needs, making it a versatile tool for the modern workplace.
Future Potential
Planned enhancements include live audio streaming for continuous transcription, multi-user support, and advanced features like improved speaker identification. Contributions are welcome to expand its capabilities and reach.

## Features
- Records 5-second audio clips from your microphone via web interface.
- Transcribes audio using the Whisper model.
- Performs speaker diarization to distinguish between speakers (requires Hugging Face token).
- Queries the xAI Grok API for summaries and action items.
- Provides text-to-speech feedback.
- Real-time communication via WebSockets.

## Installation
1. Clone the repository: `git clone https://github.com/yourusername/realtime-ai-assistant.git`
2. Navigate: `cd realtime-ai-assistant`
3. Create venv: `python3 -m venv venv`
4. Activate: `source venv/bin/activate`
5. Install: `pip install -r requirements.txt`
6. Set API keys: Create `.env` with `XAI_API_KEY=your-api-key` and optionally `HF_TOKEN=your-huggingface-token` for speaker diarization.

## Usage
- Run: `python app.py` or `uvicorn app:app --reload`
- Open browser to http://localhost:8000
- Click "Record & Analyze" to record and see results.

## Recent Updates
- Converted from Streamlit to FastAPI with WebSocket support.
- Added web-based UI with real-time audio recording.
- Enhanced real-time communication.

## Contributing
- Fork and create a feature branch.
- Commit and open a pull request.

## License
[MIT] - Consult legal.

## Additional Setup
See [docs/setup.md](docs/setup.md).
