# Architecture of Real-Time AI Meeting Assistant

## System Overview
The Real-Time AI Meeting Assistant is a Streamlit application designed to transcribe audio, summarize meetings, and extract action items using the xAI Grok API.

## Components
- **User Interface**: Built with FastAPI and WebSockets, providing a web-based interface for recording and viewing results.
- **Audio Capture**: Uses PyAudio to record 5-second audio clips from the default microphone, saved as a temporary WAV or WebM file.
- **Transcription**: Employs the Whisper model (cached) to convert audio to text with timestamps, optimized for GPU if available.
- **Speaker Diarization**: Uses pyannote.audio to identify speakers in the audio, assigning labels to transcribed segments.
- **LLM Processing**: Queries the xAI Grok API ("grok-3" model) with a structured prompt to generate JSON responses containing summaries and actions, considering speaker context.
- **Text-to-Speech**: Utilizes pyttsx3 to provide audible feedback.
- **Data Storage**: Maintains session state with a summary string and an actions DataFrame.

## Data Flow
1. User clicks "Record & Analyze" to start a 5-second recording.
2. PyAudio captures audio and saves it to `temp_audio.wav` or `temp_audio.webm`.
3. Whisper transcribes the audio into text with timestamps.
4. pyannote.audio performs speaker diarization to identify speakers.
5. Transcribed segments are labeled with speakers.
6. The diarized text is sent to the xAI API via a POST request.
7. The API response (JSON) is parsed and used to update the UI and speak the summary.
8. Actions are stored in a DataFrame for display.

## Technology Stack
- **Frontend**: FastAPI with WebSockets, HTML/CSS/JavaScript
- **Audio**: PyAudio, wave, torchaudio
- **AI**: Whisper (via openai-whisper), pyannote.audio for diarization, xAI Grok API
- **TTS**: pyttsx3
- **Data**: pandas, numpy
- **Environment**: Python 3.10+, torch (for GPU)

## Scalability and Limitations
- Current design is single-user and sequential, limiting real-time performance.
- Future enhancements could include live streaming and multi-user support.
