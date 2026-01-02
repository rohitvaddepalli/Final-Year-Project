"""
Motivation Generation Module using Ollama with LLaMA 3.2:1B

This module generates personalized motivational messages based on 
user text and detected emotion using a local LLM via Ollama.

Requirements:
    - Ollama installed and running (https://ollama.ai)
    - llama3.2:1b model pulled: ollama pull llama3.2:1b

Usage:
    from motivation_generator import MotivationGenerator
    
    generator = MotivationGenerator()
    message = generator.generate(text="I failed my exam", emotion="sad")
    print(message)
"""

import requests
import json
from typing import Optional


class MotivationGenerator:
    """
    Generate motivational messages using Ollama with LLaMA 3.2:1B.
    
    This class connects to a local Ollama instance to generate
    2-3 line motivational responses based on user emotion and text.
    """
    
    def __init__(self, model: str = "llama3.2:1b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the MotivationGenerator.
        
        Args:
            model: The Ollama model to use (default: llama3.2:1b)
            ollama_url: URL of the Ollama server (default: http://localhost:11434)
        """
        self.model = model
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
    
    def generate(self, text: str, emotion: str) -> str:
        """
        Generate a motivational message based on text and emotion.
        
        Args:
            text: The user's transcribed speech.
            emotion: Detected emotion (e.g., happy, sad, angry, fear, neutral).
        
        Returns:
            A motivational message (2-3 sentences).
        """
        # Clean inputs
        text = text.strip() if text else ""
        emotion = emotion.strip().lower() if emotion else "neutral"
        
        # Build the prompt
        prompt = self._build_prompt(text, emotion)
        
        # Call Ollama API
        response = self._call_ollama(prompt)
        
        return response if response else "Stay positive! You have the strength to overcome any challenge."
    
    def _build_prompt(self, text: str, emotion: str) -> str:
        """
        Build the prompt for the LLM.
        
        Args:
            text: User's speech text.
            emotion: Detected emotion.
        
        Returns:
            Formatted prompt string.
        """
        prompt = f"""You are a supportive and empathetic assistant. Generate a short motivational message.

User's emotion: {emotion.upper()}
User said: "{text if text else 'feeling this way'}"

Instructions:
- Write exactly 2-3 simple, supportive sentences
- Acknowledge their feelings
- Be encouraging and positive
- Do NOT give medical advice
- Reply with ONLY the motivational message, nothing else

Motivational message:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Call the Ollama API to generate a response.
        
        Args:
            prompt: The prompt to send to the model.
        
        Returns:
            Generated text or None if failed.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 100  # Limit output length
                }
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                return generated_text
            else:
                print(f"Ollama API error: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to Ollama. Make sure Ollama is running.")
            return None
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None


def motivation_generation_module(emotion_label: str, transcribed_text: str) -> str:
    """
    Main function to integrate with emotion detection module.
    
    Args:
        emotion_label: Output from emotion detection module.
        transcribed_text: Output from speech-to-text module.
    
    Returns:
        Motivational message string.
    """
    generator = MotivationGenerator()
    motivation = generator.generate(transcribed_text, emotion_label)
    
    print(f"Detected Emotion: {emotion_label}")
    print(f"Generated Motivation: {motivation}")
    
    return motivation


# =============================================================================
# MAIN: Demo and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Motivation Generator - Ollama + LLaMA 3.2:1B")
    print("=" * 60)
    
    # Initialize generator
    generator = MotivationGenerator()
    
    text = "I got job at amazon"
    emotion = "excited"
    
    print("\nGenerating motivational messages...\n")
    
    print(f"Input: \"{text}\"")
    print(f"Emotion: {emotion}")
    message = generator.generate(text, emotion)
    print(f"Response: {message}")
    print("-" * 60)
    
    print("\nDemo complete!")