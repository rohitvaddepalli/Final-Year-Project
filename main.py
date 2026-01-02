"""
Real-Time Speech Emotion Recognition and Personalized Motivation Generation System

This is the main entry point that integrates:
1. Speech Recording (via microphone)
2. Speech-to-Text (using Whisper)
3. Emotion Detection (using transformers)
4. Motivation Generation (using Ollama + LLaMA)
5. Voice Output (using Edge TTS)

Requirements:
    pip install whisper sounddevice scipy transformers pygame edge-tts requests numpy

Also requires Ollama running locally with llama3.2:1b model:
    ollama pull llama3.2:1b
"""

import os
import asyncio
import tempfile
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from transformers import pipeline

# Import custom modules
from motivation_generator import MotivationGenerator

# Try to import edge-tts for neural voices
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️  edge-tts not installed. Install with: pip install edge-tts")

# Try to import pygame for audio playback
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not installed. Install with: pip install pygame")
except Exception as e:
    PYGAME_AVAILABLE = False
    print(f"⚠️  pygame mixer init failed: {e}")


# =========================
# CONFIGURATION
# =========================
SAMPLE_RATE = 16000      # Whisper works best with 16kHz
RECORD_SECONDS = 5       # Duration to record (adjustable)
TEMP_AUDIO_FILE = "recorded_audio.wav"
VOICE_NAME = "en-US-AriaNeural"  # Neural voice for output


# =========================
# VOICE OUTPUT CLASS
# =========================
class VoiceOutput:
    """High-quality neural voice output using Microsoft Edge TTS."""
    
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", pitch: str = "+0Hz"):
        """
        Initialize the voice output system.
        
        Args:
            voice: The neural voice to use
            rate: Speech rate adjustment
            pitch: Pitch adjustment
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.temp_dir = tempfile.gettempdir()
    
    async def _speak_async(self, text: str) -> str:
        """Generate speech audio file asynchronously."""
        output_file = os.path.join(self.temp_dir, "motivation_speech.mp3")
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        await communicate.save(output_file)
        return output_file
    
    def _play_audio_pygame(self, audio_file: str):
        """Play audio file using pygame."""
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    
    def speak(self, text: str) -> bool:
        """
        Speak the given text using neural TTS.
        
        Args:
            text: Text to speak.
            
        Returns:
            True if successful, False otherwise.
        """
        if not EDGE_TTS_AVAILABLE:
            print("❌ Error: edge-tts is not installed.")
            return False
        
        if not PYGAME_AVAILABLE:
            print("❌ Error: pygame is not available.")
            return False
        
        try:
            print(f"\n🎤 Generating speech with {self.voice}...")
            
            # Generate audio file
            audio_file = asyncio.run(self._speak_async(text))
            
            print("🔊 Playing audio...")
            
            # Play the audio using pygame
            self._play_audio_pygame(audio_file)
            
            # Clean up
            pygame.mixer.music.unload()
            try:
                os.remove(audio_file)
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ Error during speech synthesis: {e}")
            return False


# =========================
# SPEECH RECORDING
# =========================
def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from microphone for specified duration."""
    print(f"\n🎤 Recording will start in 2 seconds...")
    print(f"📢 Speak for {duration} seconds after the prompt!")
    
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


# =========================
# LOAD MODELS
# =========================
def load_models():
    """Load Whisper and emotion detection models."""
    print("\n" + "=" * 60)
    print("📦 Loading Models...")
    print("=" * 60)
    
    print("\n🔄 Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    
    print("🔄 Loading Emotion Detection model...")
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    
    print("✅ Models loaded successfully!")
    
    return whisper_model, emotion_model


# =========================
# SPEECH TO TEXT
# =========================
def speech_to_text(whisper_model, audio_path):
    """Convert speech audio to text using Whisper."""
    result = whisper_model.transcribe(audio_path, language="en")
    return result["text"].strip()


# =========================
# EMOTION DETECTION
# =========================
def detect_emotion(emotion_model, text):
    """Detect emotion from text."""
    predictions = emotion_model(text)[0]
    emotion = max(predictions, key=lambda x: x["score"])
    return emotion["label"], emotion["score"], predictions


# =========================
# CLEANUP
# =========================
def cleanup(audio_file_path):
    """Delete temporary audio file."""
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
        print(f"\n🗑️ Temporary audio file '{audio_file_path}' deleted.")


# =========================
# MAIN EXECUTION
# =========================
def main():
    """Main function that runs the complete pipeline."""
    
    print("\n" + "=" * 60)
    print("🎯 Real-Time Speech Emotion Recognition &")
    print("   Personalized Motivation Generation System")
    print("=" * 60)
    
    # Step 1: Load all models
    whisper_model, emotion_model = load_models()
    
    # Step 2: Initialize motivation generator (Ollama + LLaMA)
    print("\n🔄 Initializing Motivation Generator (Ollama)...")
    motivation_generator = MotivationGenerator()
    
    # Step 3: Initialize voice output
    voice_output = VoiceOutput(
        voice=VOICE_NAME,
        rate="+0%",
        pitch="+0Hz"
    )
    
    print("\n✅ All systems ready!")
    print("\n" + "=" * 60)
    
    # Step 4: Record speech from microphone
    audio_file_path = record_audio()
    
    # Step 5: Transcribe audio to text
    print("\n📝 Transcribing audio...")
    text = speech_to_text(whisper_model, audio_file_path)
    
    if text == "":
        print("❌ No speech detected in audio. Please try again.")
        cleanup(audio_file_path)
        return
    
    # Step 6: Detect emotion from transcribed text
    emotion, confidence, scores = detect_emotion(emotion_model, text)
    
    # Display transcription and emotion results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📝 Transcribed Text:")
    print(f"   \"{text}\"")
    
    print(f"\n📊 Emotion Scores:")
    for s in scores:
        bar = "█" * int(s['score'] * 20)
        print(f"   {s['label']:12} {bar} {s['score']:.2%}")
    
    print(f"\n🎯 Detected Emotion: {emotion.upper()}")
    print(f"🔍 Confidence: {confidence:.2%}")
    
    # Step 7: Generate motivational message
    print("\n" + "=" * 60)
    print("💬 GENERATING MOTIVATION")
    print("=" * 60)
    
    print("\n⏳ Generating personalized motivational message...")
    motivation = motivation_generator.generate(text=text, emotion=emotion)
    
    print(f"\n💬 Motivational Message:")
    print(f"   {motivation}")
    
    # Step 8: Speak the motivational message
    print("\n" + "=" * 60)
    print("🔊 VOICE OUTPUT")
    print("=" * 60)
    
    if EDGE_TTS_AVAILABLE and PYGAME_AVAILABLE:
        voice_output.speak(motivation)
        print("\n✅ Voice output complete!")
    else:
        print("\n⚠️  Voice output not available. Install dependencies:")
        print("    pip install edge-tts pygame")
    
    # Step 9: Cleanup temporary files
    cleanup(audio_file_path)
    
    print("\n" + "=" * 60)
    print("🎉 Session Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
