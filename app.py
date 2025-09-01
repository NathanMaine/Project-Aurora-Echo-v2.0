from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pyaudio
import wave
import whisper
import requests
import pyttsx3
import pandas as pd
import numpy as np
import os
import torch
import json
from dotenv import load_dotenv
import asyncio
import base64
import io
from pyannote.audio import Pipeline
import torchaudio

# Load environment variables
load_dotenv()

# --- Global Constants & Initialization ---
API_KEY = os.getenv("XAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # For pyannote.audio
if not API_KEY:
    raise ValueError("API_KEY not found in .env file. Please set XAI_API_KEY.")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found. Speaker diarization will be disabled.")
API_URL = "https://api.x.ai/v1/chat/completions"
MODEL_RATE = 16000
FILENAME_WAV = "temp_audio.wav"
FILENAME_WEBM = "temp_audio.webm"

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Cache expensive resources ---
whisper_model = None
tts_engine = None
diarization_pipeline = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    return whisper_model

def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 180)
    return tts_engine

def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None and HF_TOKEN:
        try:
            print("Loading speaker diarization pipeline...")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
            if torch.cuda.is_available():
                diarization_pipeline.to(torch.device("cuda"))
            print("Speaker diarization pipeline loaded successfully.")
        except Exception as e:
            print(f"Failed to load diarization pipeline: {e}")
            print("Speaker diarization will be disabled.")
            diarization_pipeline = None
    return diarization_pipeline

# --- Core Functions ---
async def process_audio(audio_data: bytes):
    """Processes audio data and returns transcription and LLM response."""
    try:
        print(f"Received audio data of length: {len(audio_data)} bytes")
        
        # Check if it's a WAV file (starts with RIFF)
        if len(audio_data) > 12 and audio_data[:4] == b'RIFF':
            print("Detected WAV format")
            filename = FILENAME_WAV
            # Save as WAV
            with open(filename, 'wb') as f:
                f.write(audio_data)
        else:
            print("Detected non-WAV format, saving as WebM")
            filename = FILENAME_WEBM
            # Save as WebM
            with open(filename, 'wb') as f:
                f.write(audio_data)
        
        print("Audio file saved, loading Whisper model...")
        
        # Transcribe with segments
        model = get_whisper_model()
        result = model.transcribe(filename, fp16=False)
        transcribed_text = result.get("text", "").strip()
        segments = result.get("segments", [])
        
        print(f"Transcription result: '{transcribed_text}'")
        
        # Perform speaker diarization if available
        diarized_text = transcribed_text
        if HF_TOKEN and segments:
            pipeline = get_diarization_pipeline()
            if pipeline is not None:
                try:
                    # Load audio for diarization
                    waveform, sample_rate = torchaudio.load(filename)
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    
                    # Perform diarization
                    diarization = pipeline({"waveform": waveform, "sample_rate": 16000})
                    
                    # Assign speakers to segments
                    diarized_segments = []
                    for segment in segments:
                        start = segment['start']
                        end = segment['end']
                        text = segment['text'].strip()
                        if text:
                            # Find the speaker for this segment
                            speakers = []
                            for turn, _, speaker in diarization.itertracks(yield_label=True):
                                if turn.start <= start < turn.end or turn.start < end <= turn.end or (start <= turn.start and end >= turn.end):
                                    speakers.append(speaker)
                            speaker = speakers[0] if speakers else "Unknown"
                            diarized_segments.append(f"{speaker}: {text}")
                    
                    diarized_text = " ".join(diarized_segments)
                    print(f"Diarized text: '{diarized_text}'")
                except Exception as e:
                    print(f"Diarization error: {e}")
                    print("Falling back to regular transcription without speaker labels.")
                    # Fall back to original text
            else:
                print("Diarization pipeline not available, using regular transcription.")
        
        if not diarized_text:
            return {"error": "No speech detected. Please speak clearly and ensure your microphone is working."}

        # Query LLM
        llm_response = await query_llm(diarized_text)
        if llm_response:
            return {
                "transcription": diarized_text,
                "summary": llm_response.get("summary", ""),
                "actions": llm_response.get("actions", [])
            }
        else:
            return {"error": "Failed to get LLM response."}

    except Exception as e:
        print(f"Error in process_audio: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Processing error: {str(e)}"}
    finally:
        for f in [FILENAME_WAV, FILENAME_WEBM]:
            if os.path.exists(f):
                os.remove(f)

async def query_llm(text):
    """Sends transcription to the LLM and returns the parsed response."""
    prompt = f'You are a meeting assistant. The following text includes speaker labels (e.g., SPEAKER_00, SPEAKER_01). Summarize the meeting content, considering different speakers for richer context, and extract action items. Format your response as a JSON object with two keys: "summary" and "actions". The "actions" array should contain objects with "task", "assignee", and "due" keys.\n\nText: {text}'
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        llm_output_str = data['choices'][0]['message']['content']
        return json.loads(llm_output_str)
    except Exception as e:
        print(f"API Error: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WebSocket data: {data[:100]}...")  # Log first 100 chars
            if data.startswith("audio:"):
                # Decode base64 audio data
                audio_b64 = data[6:]
                print(f"Audio base64 length: {len(audio_b64)}")
                audio_data = base64.b64decode(audio_b64)
                print(f"Decoded audio length: {len(audio_data)} bytes")
                result = await process_audio(audio_data)
                print(f"Processing result: {result}")
                await websocket.send_json(result)
                # Optionally, speak the summary
                if "summary" in result:
                    engine = get_tts_engine()
                    engine.say(result["summary"])
                    engine.runAndWait()
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
