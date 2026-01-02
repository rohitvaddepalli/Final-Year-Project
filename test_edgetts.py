"""
Test script for Motivation Generator with Realistic Voice Output

Uses edge-tts for high-quality neural text-to-speech synthesis.
Microsoft Edge TTS provides natural, realistic voices.

Install required packages:
    pip install edge-tts pygame

Available voices (examples):
    - en-US-AriaNeural (Female, warm and expressive)
    - en-US-GuyNeural (Male, friendly)
    - en-US-JennyNeural (Female, conversational)
    - en-GB-SoniaNeural (British Female)
    - en-IN-NeerjaNeural (Indian Female)
    - en-IN-PrabhatNeural (Indian Male)
"""

import asyncio
import os
import tempfile
import time
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


class VoiceOutput:
    """
    High-quality neural voice output using Microsoft Edge TTS.
    
    Provides realistic, natural-sounding speech synthesis.
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", pitch: str = "+0Hz"):
        """
        Initialize the voice output system.
        
        Args:
            voice: The neural voice to use (default: en-US-AriaNeural)
            rate: Speech rate adjustment (e.g., "+10%" for faster, "-10%" for slower)
            pitch: Pitch adjustment (e.g., "+5Hz" for higher)
        
        Available voices (run list_voices() to see all):
            - en-US-AriaNeural: Female, expressive and warm
            - en-US-GuyNeural: Male, friendly and casual
            - en-US-JennyNeural: Female, conversational
            - en-US-ChristopherNeural: Male, professional
            - en-GB-SoniaNeural: British Female, clear
            - en-IN-NeerjaNeural: Indian Female
            - en-IN-PrabhatNeural: Indian Male
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.temp_dir = tempfile.gettempdir()
    
    async def _speak_async(self, text: str) -> str:
        """
        Generate speech audio file asynchronously.
        
        Args:
            text: Text to convert to speech.
            
        Returns:
            Path to the generated audio file.
        """
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
        """
        Play audio file using pygame.
        
        Args:
            audio_file: Path to the audio file.
        """
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
            print("   Install with: pip install edge-tts")
            return False
        
        if not PYGAME_AVAILABLE:
            print("❌ Error: pygame is not available.")
            print("   Install with: pip install pygame")
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
    
    @staticmethod
    async def list_voices_async():
        """List all available voices."""
        voices = await edge_tts.list_voices()
        return voices
    
    @staticmethod
    def list_voices(language_filter: str = "en"):
        """
        Print available voices, optionally filtered by language.
        
        Args:
            language_filter: Language code prefix to filter (e.g., "en" for English)
        """
        if not EDGE_TTS_AVAILABLE:
            print("edge-tts not installed")
            return
        
        voices = asyncio.run(VoiceOutput.list_voices_async())
        
        print(f"\n{'='*60}")
        print(f"Available Voices (filtered: {language_filter})")
        print(f"{'='*60}\n")
        
        for voice in voices:
            if voice["Locale"].startswith(language_filter):
                print(f"  Voice: {voice['ShortName']}")
                print(f"    Gender: {voice['Gender']}")
                print(f"    Locale: {voice['Locale']}")
                print()


def main():
    """Main function demonstrating motivation generation with voice output."""
    
    print("=" * 60)
    print("🎯 Motivation Generator with Realistic Voice Output")
    print("=" * 60)
    
    # Initialize the motivation generator
    generator = MotivationGenerator()
    
    # Initialize voice output with a warm, expressive voice
    # You can change the voice to any from the list above
    voice = VoiceOutput(
        voice="en-US-GuyNeural",  # Expressive male voice
        rate="+2%",                 # Slightly slower for warmth
        pitch="+2Hz"                # Natural pitch
    )
    
    # Generate motivation
    text = "I got a job at google and i am so excited"
    emotion = "excited"  # Changed to happy since promotion is good news!
    
    print(f"\n📝 Input Text: \"{text}\"")
    print(f"😊 Detected Emotion: {emotion}")
    print("\n⏳ Generating motivational message...")
    
    message = generator.generate(text=text, emotion=emotion)
    
    print(f"\n💬 Response: {message}")
    
    # Speak the response with realistic voice
    if EDGE_TTS_AVAILABLE and PYGAME_AVAILABLE:
        voice.speak(message)
        print("\n✅ Done!")
    else:
        print("\n⚠️  Voice output not available. Install dependencies:")
        print("    pip install edge-tts pygame")


if __name__ == "__main__":
    # Uncomment the line below to see all available English voices
    # VoiceOutput.list_voices("en")
    
    main()