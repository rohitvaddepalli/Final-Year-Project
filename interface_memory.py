"""
Interface and Memory Module for Speech Emotion Recognition System

This module provides:
1. A graphical user interface (GUI) using Tkinter for easy interaction
2. A conversation memory system to track past interactions, emotions, and responses

Requirements:
    pip install tkinter (usually included with Python)
    
Usage:
    python interface_memory.py
"""

import os
import json
import asyncio
import tempfile
import time
import threading
import shutil
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# GUI imports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Helper to check for system commands
def is_ffmpeg_installed():
    return shutil.which("ffmpeg") is not None

# Audio imports
MISSING_DEPS = []

try:
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    MISSING_DEPS.append("sounddevice/scipy/numpy (pip install sounddevice scipy numpy)")

# Whisper for speech-to-text
try:
    import whisper
    WHISPER_PACKAGE_AVAILABLE = True
except ImportError:
    WHISPER_PACKAGE_AVAILABLE = False
    MISSING_DEPS.append("whisper (pip install openai-whisper)")

FFMPEG_AVAILABLE = is_ffmpeg_installed()
if not FFMPEG_AVAILABLE:
    MISSING_DEPS.append("FFmpeg (System dependency - download from ffmpeg.org)")

WHISPER_AVAILABLE = WHISPER_PACKAGE_AVAILABLE and FFMPEG_AVAILABLE

# Emotion detection
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    MISSING_DEPS.append("transformers/torch (pip install transformers torch)")

# Voice output
try:
    import edge_tts
    import pygame
    pygame.mixer.init()
    TTS_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    MISSING_DEPS.append("edge_tts/pygame (pip install edge-tts pygame)")

# Import motivation generator
try:
    from motivation_generator import MotivationGenerator
    MOTIVATION_AVAILABLE = True
except ImportError:
    MOTIVATION_AVAILABLE = False
    MISSING_DEPS.append("motivation_generator.py (missing file)")


# =============================================================================
# MEMORY SYSTEM
# =============================================================================

@dataclass
class ConversationEntry:
    """Represents a single conversation/interaction entry."""
    timestamp: str
    user_text: str
    detected_emotion: str
    emotion_confidence: float
    emotion_scores: Dict[str, float]
    motivation_response: str
    session_id: str


class ConversationMemory:
    """
    Manages conversation history and provides memory persistence.
    
    Features:
    - Store conversation entries with timestamps
    - Save/load history to/from JSON file
    - Query past conversations by emotion, date, etc.
    - Track emotion patterns over time
    """
    
    def __init__(self, memory_file: str = "conversation_history.json"):
        """
        Initialize the conversation memory.
        
        Args:
            memory_file: Path to the JSON file for persistence.
        """
        self.memory_file = memory_file
        self.history: List[ConversationEntry] = []
        self.current_session_id = self._generate_session_id()
        self._load_history()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _load_history(self):
        """Load conversation history from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = [
                        ConversationEntry(**entry) for entry in data
                    ]
                print(f"✅ Loaded {len(self.history)} conversation entries from memory.")
            except Exception as e:
                print(f"⚠️ Error loading memory: {e}")
                self.history = []
        else:
            self.history = []
    
    def save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(entry) for entry in self.history], f, indent=2)
            print(f"💾 Saved {len(self.history)} conversation entries to memory.")
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
    def add_entry(self, user_text: str, emotion: str, confidence: float,
                  scores: Dict[str, float], motivation: str):
        """
        Add a new conversation entry.
        
        Args:
            user_text: The user's transcribed speech.
            emotion: Detected emotion label.
            confidence: Confidence score for the emotion.
            scores: All emotion scores.
            motivation: Generated motivational response.
        """
        entry = ConversationEntry(
            timestamp=datetime.now().isoformat(),
            user_text=user_text,
            detected_emotion=emotion,
            emotion_confidence=confidence,
            emotion_scores=scores,
            motivation_response=motivation,
            session_id=self.current_session_id
        )
        self.history.append(entry)
        self.save_history()
        return entry
    
    def get_recent_entries(self, count: int = 5) -> List[ConversationEntry]:
        """Get the most recent conversation entries."""
        return self.history[-count:] if self.history else []
    
    def get_entries_by_emotion(self, emotion: str) -> List[ConversationEntry]:
        """Get all entries with a specific emotion."""
        return [e for e in self.history if e.detected_emotion.lower() == emotion.lower()]
    
    def get_emotion_statistics(self) -> Dict[str, int]:
        """Get statistics on emotion frequency."""
        stats = {}
        for entry in self.history:
            emotion = entry.detected_emotion.lower()
            stats[emotion] = stats.get(emotion, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def get_session_entries(self, session_id: Optional[str] = None) -> List[ConversationEntry]:
        """Get all entries from a specific session or current session."""
        sid = session_id or self.current_session_id
        return [e for e in self.history if e.session_id == sid]
    
    def clear_history(self):
        """Clear all conversation history."""
        self.history = []
        self.save_history()
    
    def get_context_for_llm(self, num_entries: int = 3) -> str:
        """
        Get recent conversation context formatted for LLM input.
        
        This can be used to provide context to the motivation generator
        for more personalized responses.
        """
        recent = self.get_recent_entries(num_entries)
        if not recent:
            return ""
        
        context = "Recent conversation history:\n"
        for entry in recent:
            context += f"- User said: \"{entry.user_text}\" (feeling: {entry.detected_emotion})\n"
            context += f"  Response: \"{entry.motivation_response}\"\n"
        
        return context


# =============================================================================
# GUI INTERFACE
# =============================================================================

class EmotionRecognitionGUI:
    """
    Modern graphical user interface for the Speech Emotion Recognition System.
    
    Features:
    - Record and analyze speech
    - Display emotion detection results
    - Show conversation history
    - Visualize emotion statistics
    """
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI."""
        self.root = root
        self.root.title("🎯 Speech Emotion Recognition System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Set theme colors
        self.colors = {
            'bg': '#1a1a2e',
            'secondary_bg': '#16213e',
            'accent': '#0f3460',
            'highlight': '#e94560',
            'text': '#eaeaea',
            'success': '#00d4aa',
            'warning': '#ffc857',
            'error': '#ff6b6b'
        }
        
        # Configure root background
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize components
        self.memory = ConversationMemory()
        self.models_loaded = False
        self.whisper_model = None
        self.emotion_model = None
        self.motivation_generator = None
        self.voice_output = None
        self.is_recording = False
        
        # Build the UI
        self._create_styles()
        self._create_widgets()
        
        # Start model loading in background
        self._load_models_async()
    
    def _create_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button style
        style.configure('Accent.TButton',
                       background=self.colors['highlight'],
                       foreground='white',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(20, 10))
        
        style.map('Accent.TButton',
                 background=[('active', '#c73e54'), ('disabled', '#555555')])
        
        # Configure label style
        style.configure('Header.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Info.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🎯 Speech Emotion Recognition & Motivation System",
            font=('Segoe UI', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        status_container = tk.Frame(header_frame, bg=self.colors['bg'])
        status_container.pack(side=tk.RIGHT)

        self.status_label = tk.Label(
            status_container,
            text="⏳ Loading models...",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['warning']
        )
        self.status_label.pack(side=tk.TOP, anchor=tk.E)

        self.dep_btn = tk.Button(
            status_container,
            text="🔍 Check Dependencies",
            font=('Segoe UI', 8),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=5,
            pady=2,
            command=self._show_dependency_status,
            cursor='hand2'
        )
        self.dep_btn.pack(side=tk.TOP, anchor=tk.E, pady=(2, 0))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Main Interface
        self.main_tab = tk.Frame(self.notebook, bg=self.colors['secondary_bg'])
        self.notebook.add(self.main_tab, text="  🎤 Record & Analyze  ")
        self._create_main_tab()
        
        # Tab 2: History
        self.history_tab = tk.Frame(self.notebook, bg=self.colors['secondary_bg'])
        self.notebook.add(self.history_tab, text="  📜 History  ")
        self._create_history_tab()
        
        # Tab 3: Statistics
        self.stats_tab = tk.Frame(self.notebook, bg=self.colors['secondary_bg'])
        self.notebook.add(self.stats_tab, text="  📊 Statistics  ")
        self._create_stats_tab()
    
    def _create_main_tab(self):
        """Create the main recording/analysis tab."""
        # Warning banner for missing dependencies
        self.warning_banner = tk.Label(
            self.main_tab,
            text="",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['error'],
            fg='white',
            pady=5
        )
        self._update_warning_banner()

        # Recording section
        record_frame = tk.Frame(self.main_tab, bg=self.colors['accent'], padx=20, pady=20)
        record_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Recording button
        self.record_btn = tk.Button(
            record_frame,
            text="🎤 Start Recording",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['highlight'],
            fg='white',
            activebackground='#c73e54',
            activeforeground='white',
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor='hand2',
            command=self._toggle_recording
        )
        self.record_btn.pack(pady=10)
        
        # Recording status
        self.recording_status = tk.Label(
            record_frame,
            text="Click to record (Stops automatically when you're done speaking)",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg=self.colors['text']
        )
        self.recording_status.pack(pady=5)
        
        # Results section
        results_frame = tk.Frame(self.main_tab, bg=self.colors['secondary_bg'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Transcription
        trans_label = tk.Label(
            results_frame,
            text="📝 Transcribed Text:",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['secondary_bg'],
            fg=self.colors['text']
        )
        trans_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.transcription_text = tk.Text(
            results_frame,
            height=3,
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        self.transcription_text.pack(fill=tk.X, pady=(0, 10))
        
        # Emotion display
        emotion_frame = tk.Frame(results_frame, bg=self.colors['secondary_bg'])
        emotion_frame.pack(fill=tk.X, pady=10)
        
        emotion_label = tk.Label(
            emotion_frame,
            text="🎯 Detected Emotion:",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['secondary_bg'],
            fg=self.colors['text']
        )
        emotion_label.pack(side=tk.LEFT)
        
        self.emotion_display = tk.Label(
            emotion_frame,
            text="---",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['secondary_bg'],
            fg=self.colors['highlight']
        )
        self.emotion_display.pack(side=tk.LEFT, padx=10)
        
        self.confidence_display = tk.Label(
            emotion_frame,
            text="",
            font=('Segoe UI', 10),
            bg=self.colors['secondary_bg'],
            fg=self.colors['success']
        )
        self.confidence_display.pack(side=tk.LEFT)
        
        # Motivation response
        motiv_label = tk.Label(
            results_frame,
            text="💬 Motivational Response:",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['secondary_bg'],
            fg=self.colors['text']
        )
        motiv_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.motivation_text = tk.Text(
            results_frame,
            height=4,
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg=self.colors['success'],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        self.motivation_text.pack(fill=tk.X, pady=(0, 10))
        
        # Speak button
        self.speak_btn = tk.Button(
            results_frame,
            text="🔊 Speak Response",
            font=('Segoe UI', 11),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            activebackground=self.colors['highlight'],
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2',
            command=self._speak_motivation,
            state=tk.DISABLED
        )
        self.speak_btn.pack(pady=10)
    
    def _create_history_tab(self):
        """Create the conversation history tab."""
        # History list
        history_frame = tk.Frame(self.history_tab, bg=self.colors['secondary_bg'])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        header = tk.Label(
            history_frame,
            text="📜 Conversation History",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['secondary_bg'],
            fg=self.colors['text']
        )
        header.pack(anchor=tk.W, pady=(0, 10))
        
        # Scrollable history
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            font=('Consolas', 10),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        btn_frame = tk.Frame(history_frame, bg=self.colors['secondary_bg'])
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        refresh_btn = tk.Button(
            btn_frame,
            text="🔄 Refresh",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._refresh_history
        )
        refresh_btn.pack(side=tk.LEFT)
        
        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Clear History",
            font=('Segoe UI', 10),
            bg=self.colors['error'],
            fg='white',
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._clear_history
        )
        clear_btn.pack(side=tk.RIGHT)
        
        # Load initial history
        self._refresh_history()
    
    def _create_stats_tab(self):
        """Create the statistics tab."""
        stats_frame = tk.Frame(self.stats_tab, bg=self.colors['secondary_bg'])
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        header = tk.Label(
            stats_frame,
            text="📊 Emotion Statistics",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['secondary_bg'],
            fg=self.colors['text']
        )
        header.pack(anchor=tk.W, pady=(0, 10))
        
        # Stats display
        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            font=('Consolas', 11),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            height=20
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        refresh_btn = tk.Button(
            stats_frame,
            text="🔄 Refresh Statistics",
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._refresh_stats
        )
        refresh_btn.pack(pady=(10, 0))
        
        # Load initial stats
        self._refresh_stats()
    
    def _load_models_async(self):
        """Load models in a background thread."""
        def load():
            try:
                if WHISPER_AVAILABLE:
                    self.whisper_model = whisper.load_model("base")
                
                if TRANSFORMERS_AVAILABLE:
                    self.emotion_model = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        top_k=None
                    )
                
                if MOTIVATION_AVAILABLE:
                    self.motivation_generator = MotivationGenerator()
                
                self.models_loaded = True
                self.root.after(0, self._on_models_loaded)
            except Exception as e:
                self.root.after(0, lambda: self._on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_models_loaded(self):
        """Called when models are loaded successfully."""
        if MISSING_DEPS:
            self.status_label.config(text="⚠️ Issues Found", fg=self.colors['error'])
        else:
            self.status_label.config(text="✅ Ready", fg=self.colors['success'])
        
        self.record_btn.config(state=tk.NORMAL)
        self._update_warning_banner()

    def _update_warning_banner(self):
        """Update the visibility and text of the warning banner."""
        if not WHISPER_AVAILABLE:
            msg = "⚠️ Speech-to-Text is unavailable. "
            if not WHISPER_PACKAGE_AVAILABLE:
                msg += "Missing: openai-whisper. "
            if not FFMPEG_AVAILABLE:
                msg += "Missing: FFmpeg (System Tool)."
            
            self.warning_banner.config(text=msg)
            self.warning_banner.pack(fill=tk.X, side=tk.TOP)
        else:
            self.warning_banner.pack_forget()

    def _show_dependency_status(self):
        """Show a detailed popup with dependency status."""
        status_msg = "🔍 System Dependency Status:\n\n"
        
        deps = [
            ("Python Package: openai-whisper", WHISPER_PACKAGE_AVAILABLE),
            ("System Tool: FFmpeg", FFMPEG_AVAILABLE),
            ("Python Package: sounddevice", AUDIO_AVAILABLE),
            ("Python Package: transformers", TRANSFORMERS_AVAILABLE),
            ("Python Package: edge-tts", TTS_AVAILABLE),
            ("Core File: motivation_generator.py", MOTIVATION_AVAILABLE)
        ]
        
        for name, available in deps:
            status = "✅ OK" if available else "❌ MISSING"
            status_msg += f"{status} | {name}\n"
        
        if MISSING_DEPS:
            status_msg += "\n💡 How to fix:\n"
            for dep in MISSING_DEPS:
                status_msg += f"- {dep}\n"
        else:
            status_msg += "\n✨ All systems go! You're ready to record."
            
        messagebox.showinfo("Dependency Check", status_msg)
    
    def _on_model_error(self, error: str):
        """Called when model loading fails."""
        self.status_label.config(text=f"❌ Error: {error}", fg=self.colors['error'])
        messagebox.showerror("Model Loading Error", f"Failed to load models:\n{error}")
    
    def _toggle_recording(self):
        """Start or stop recording."""
        if not self.models_loaded:
            messagebox.showwarning("Not Ready", "Please wait for models to load.")
            return
        
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio recording is not available. Install sounddevice and scipy.")
            return
        
        if not self.is_recording:
            self._start_recording()
        else:
            self.is_recording = False
    
    def _start_recording(self):
        """Start recording audio with automatic silence detection."""
        self.is_recording = True
        self.record_btn.config(text="⏹️ Stop Recording", bg=self.colors['error'])
        self.recording_status.config(text="🔴 Listening... Speak now!")
        
        def record():
            try:
                sample_rate = 16000
                chunk_size = 1024
                # RMS threshold for silence (400 is a good starting point for quiet speech)
                threshold = 400 
                # How many seconds of silence before stopping automatically
                silence_limit = 1.5 
                
                audio_buffer = []
                silent_chunks = 0
                max_silent_chunks = int(silence_limit * sample_rate / chunk_size)
                
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                    while self.is_recording:
                        chunk, overflowed = stream.read(chunk_size)
                        audio_buffer.append(chunk)
                        
                        # Calculate energy (RMS) to detect silence
                        rms = np.sqrt(np.mean(chunk.astype(float)**2))
                        
                        if rms < threshold:
                            silent_chunks += 1
                        else:
                            silent_chunks = 0
                            
                        # Automatically stop if silence limit reached
                        if silent_chunks > max_silent_chunks and len(audio_buffer) > (sample_rate / chunk_size):
                            break
                        
                        # Safety limit (max 1 minute of recording)
                        if len(audio_buffer) * chunk_size > sample_rate * 60:
                            break
                
                if audio_buffer:
                    # Combine all recorded chunks
                    audio_data = np.concatenate(audio_buffer)
                    
                    # Save to temporary WAV file
                    temp_file = "temp_recording.wav"
                    write(temp_file, sample_rate, audio_data)
                    
                    self.root.after(0, lambda: self._process_recording(temp_file))
                else:
                    self.root.after(0, lambda: self._on_recording_error("No audio captured"))
                    
            except Exception as e:
                self.root.after(0, lambda: self._on_recording_error(str(e)))
        
        thread = threading.Thread(target=record, daemon=True)
        thread.start()
    
    def _process_recording(self, audio_file: str):
        """Process the recorded audio."""
        self.record_btn.config(text="⏳ Processing...", bg=self.colors['warning'])
        self.recording_status.config(text="Analyzing your speech...")
        
        def process():
            try:
                # Transcribe
                if self.whisper_model:
                    result = self.whisper_model.transcribe(audio_file, language="en")
                    text = result["text"].strip()
                else:
                    text = "[Whisper not available]"
                
                # Detect emotion
                if self.emotion_model and text:
                    predictions = self.emotion_model(text)[0]
                    emotion = max(predictions, key=lambda x: x["score"])
                    emotion_label = emotion["label"]
                    confidence = emotion["score"]
                    scores = {p["label"]: p["score"] for p in predictions}
                else:
                    emotion_label = "neutral"
                    confidence = 0.0
                    scores = {}
                
                # Generate motivation
                if self.motivation_generator and text:
                    motivation = self.motivation_generator.generate(text, emotion_label)
                else:
                    motivation = "Stay positive! You're doing great."
                
                # Save to memory
                self.memory.add_entry(text, emotion_label, confidence, scores, motivation)
                
                # Update UI
                self.root.after(0, lambda: self._display_results(
                    text, emotion_label, confidence, motivation
                ))
                
                # Cleanup
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    
            except Exception as e:
                self.root.after(0, lambda: self._on_recording_error(str(e)))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _display_results(self, text: str, emotion: str, confidence: float, motivation: str):
        """Display analysis results and automatically speak the motivation."""
        self.record_btn.config(text="🎤 Start Recording", bg=self.colors['highlight'])
        self.recording_status.config(text="Click to record (Stops automatically when you're done speaking)")
        self.is_recording = False
        
        # Update displays
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, text or "[No speech detected]")
        
        self.emotion_display.config(text=emotion.upper())
        self.confidence_display.config(text=f"({confidence:.1%} confidence)")
        
        self.motivation_text.delete(1.0, tk.END)
        self.motivation_text.insert(tk.END, motivation)
        
        self.speak_btn.config(state=tk.NORMAL)
        
        # Refresh history
        self._refresh_history()
        self._refresh_stats()
        
        # Automatically speak the motivation response
        if TTS_AVAILABLE and motivation:
            self._auto_speak_motivation(motivation)
    
    def _on_recording_error(self, error: str):
        """Handle recording error."""
        self.record_btn.config(text="🎤 Start Recording", bg=self.colors['highlight'])
        self.recording_status.config(text="Click to record (Stops automatically when you're done speaking)")
        self.is_recording = False
        messagebox.showerror("Recording Error", f"An error occurred:\n{error}")
    
    def _auto_speak_motivation(self, motivation: str):
        """Automatically speak the motivation response after analysis."""
        def speak():
            try:
                # Update UI to show speaking status
                self.root.after(0, lambda: self.recording_status.config(
                    text="🔊 Speaking motivation...", 
                    fg=self.colors['success']
                ))
                
                temp_file = os.path.join(tempfile.gettempdir(), "motivation_speech.mp3")
                
                # Generate speech using Edge TTS with a warm, friendly voice
                async def generate():
                    communicate = edge_tts.Communicate(
                        text=motivation, 
                        voice="en-US-GuyNeural",  # Friendly male voice
                        rate="+2%",  # Slightly adjusted rate
                        pitch="+2Hz"  # Natural pitch
                    )
                    await communicate.save(temp_file)
                
                asyncio.run(generate())
                
                # Play the audio
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Cleanup
                pygame.mixer.music.unload()
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                # Reset status
                self.root.after(0, lambda: self.recording_status.config(
                    text="✅ Voice output complete! Click to record again.",
                    fg=self.colors['text']
                ))
                
            except Exception as e:
                print(f"Auto TTS Error: {e}")
                self.root.after(0, lambda: self.recording_status.config(
                    text="Click to record (Stops automatically when you're done speaking)",
                    fg=self.colors['text']
                ))
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
    
    def _speak_motivation(self):
        """Speak the motivational response."""
        if not TTS_AVAILABLE:
            messagebox.showwarning("TTS Unavailable", "Text-to-speech is not available. Install edge-tts and pygame.")
            return
        
        motivation = self.motivation_text.get(1.0, tk.END).strip()
        if not motivation:
            return
        
        def speak():
            try:
                temp_file = os.path.join(tempfile.gettempdir(), "motivation_speech.mp3")
                
                async def generate():
                    communicate = edge_tts.Communicate(text=motivation, voice="en-US-AriaNeural")
                    await communicate.save(temp_file)
                
                asyncio.run(generate())
                
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                pygame.mixer.music.unload()
                os.remove(temp_file)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("TTS Error", str(e)))
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
    
    def _refresh_history(self):
        """Refresh the history display."""
        self.history_text.delete(1.0, tk.END)
        
        entries = self.memory.get_recent_entries(20)
        
        if not entries:
            self.history_text.insert(tk.END, "No conversation history yet.\n\n")
            self.history_text.insert(tk.END, "Start recording to build your history!")
            return
        
        for i, entry in enumerate(reversed(entries), 1):
            timestamp = datetime.fromisoformat(entry.timestamp).strftime("%Y-%m-%d %H:%M")
            
            self.history_text.insert(tk.END, f"{'─' * 50}\n")
            self.history_text.insert(tk.END, f"📅 {timestamp} | Session: {entry.session_id}\n")
            self.history_text.insert(tk.END, f"🎯 Emotion: {entry.detected_emotion.upper()} ({entry.emotion_confidence:.1%})\n")
            self.history_text.insert(tk.END, f"📝 Said: \"{entry.user_text}\"\n")
            self.history_text.insert(tk.END, f"💬 Response: \"{entry.motivation_response}\"\n\n")
    
    def _refresh_stats(self):
        """Refresh the statistics display."""
        self.stats_text.delete(1.0, tk.END)
        
        stats = self.memory.get_emotion_statistics()
        total = sum(stats.values())
        
        self.stats_text.insert(tk.END, "═" * 40 + "\n")
        self.stats_text.insert(tk.END, "       EMOTION FREQUENCY ANALYSIS\n")
        self.stats_text.insert(tk.END, "═" * 40 + "\n\n")
        
        self.stats_text.insert(tk.END, f"📊 Total Conversations: {total}\n")
        self.stats_text.insert(tk.END, f"📅 Current Session: {self.memory.current_session_id}\n\n")
        
        if stats:
            self.stats_text.insert(tk.END, "📈 Emotion Distribution:\n\n")
            
            for emotion, count in stats.items():
                percentage = (count / total) * 100
                bar_length = int(percentage / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                emoji = {
                    'joy': '😊', 'sadness': '😢', 'anger': '😠',
                    'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
                    'neutral': '😐'
                }.get(emotion.lower(), '🔹')
                
                self.stats_text.insert(
                    tk.END,
                    f"  {emoji} {emotion.capitalize():12} [{bar}] {count:3} ({percentage:5.1f}%)\n"
                )
        else:
            self.stats_text.insert(tk.END, "No data yet. Start recording to collect statistics!")
    
    def _clear_history(self):
        """Clear conversation history."""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all history?"):
            self.memory.clear_history()
            self._refresh_history()
            self._refresh_stats()
            messagebox.showinfo("Cleared", "Conversation history has been cleared.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_gui():
    """Run the GUI application."""
    root = tk.Tk()
    app = EmotionRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Speech Emotion Recognition - Interface & Memory Module")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    print(f"  ✓ Audio Recording: {'Available' if AUDIO_AVAILABLE else 'Not Available'}")
    print(f"  ✓ Whisper STT: {'Available' if WHISPER_AVAILABLE else 'Not Available'}")
    print(f"  ✓ Transformers: {'Available' if TRANSFORMERS_AVAILABLE else 'Not Available'}")
    print(f"  ✓ TTS Output: {'Available' if TTS_AVAILABLE else 'Not Available'}")
    print(f"  ✓ Motivation Gen: {'Available' if MOTIVATION_AVAILABLE else 'Not Available'}")
    
    print("\n🚀 Starting GUI...")
    run_gui()
