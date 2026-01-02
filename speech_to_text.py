import os
import whisper
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline

# =========================
# RECORDING SETTINGS
# =========================
SAMPLE_RATE = 16000  # Whisper works best with 16kHz
RECORD_SECONDS = 5   # Duration to record (adjustable)
TEMP_AUDIO_FILE = "recorded_audio.wav"

# =========================
# RECORD AUDIO FROM MICROPHONE
# =========================
def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from microphone for specified duration."""
    print(f"\n🎤 Recording will start in 2 seconds...")
    print(f"📢 Speak for {duration} seconds after the beep!")
    
    # Small delay to prepare
    sd.sleep(2000)
    
    print("\n🔴 RECORDING... Speak now!")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()  # Wait until recording is complete
    
    print("✅ Recording complete!")
    
    # Save to WAV file
    write(TEMP_AUDIO_FILE, sample_rate, audio_data)
    print(f"💾 Audio saved to: {TEMP_AUDIO_FILE}")
    
    return TEMP_AUDIO_FILE

# Record speech from microphone
audio_file_path = record_audio()
print("Using recorded audio:", audio_file_path)

# =========================
# LOAD MODELS
# =========================
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading Emotion Detection model...")
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None   # ✅ replaces return_all_scores=True
)

# =========================
# AUDIO → TEXT
# =========================
def speech_to_text(audio_path):
    result = whisper_model.transcribe(audio_path, language="en")
    return result["text"].strip()

# =========================
# TEXT → EMOTION
# =========================
def detect_emotion(text):
    predictions = emotion_model(text)[0]
    emotion = max(predictions, key=lambda x: x["score"])
    return emotion["label"], emotion["score"], predictions

# =========================
# MAIN EXECUTION
# =========================
print("\nProcessing audio file...")
text = speech_to_text(audio_file_path)

if text == "":
    print("❌ No speech detected in audio")
else:
    emotion, confidence, scores = detect_emotion(text)

    print("\n==============================")
    print("📝 Transcribed Text:")
    print(text)

    print("\n📊 Emotion Scores:")
    for s in scores:
        print(f"{s['label']}: {s['score']:.4f}")

    print("\n🎯 Final Emotion:", emotion)
    print("🔍 Confidence:", round(confidence, 4))
    print("==============================")

# =========================
# CLEANUP: DELETE TEMP AUDIO FILE
# =========================
if os.path.exists(audio_file_path):
    os.remove(audio_file_path)
    print(f"\n🗑️ Temporary audio file '{audio_file_path}' deleted.")
