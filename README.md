# Real-Time Speech Emotion Recognition & Personalized Motivation System 🎯

An intelligent, real-time system designed to provide emotional support by recognizing human emotions from speech and generating personalized motivational responses.

## 🌟 Features
- **Real-Time Speech Recognition:** Converts live voice input to text using OpenAI's Whisper.
- **Emotion Detection:** Analyzes the semantic meaning of speech to classify emotions (Joy, Sadness, Anger, Fear, Neutral) using DistilRoBERTa.
- **Personalized Motivation:** Generates empathetic, context-aware motivational messages based on the detected emotion.
- **Voice Output:** Delivers responses back to the user through a natural-sounding neural voice (Edge-TTS).
- **Conversation Memory:** Tracks interaction history and provides emotion frequency analytics.
- **Privacy-First:** Designed to work offline to ensure user data security and low latency.

## 🛠️ Technology Stack
- **Core:** Python 3.12+
- **GUI:** Tkinter
- **STT:** OpenAI Whisper (Base Model)
- **Emotion Analysis:** j-hartmann/emotion-english-distilroberta-base
- **Response Generation:** LLaMA 3 Integrated / Rule-based fallback
- **Voice:** Microsoft Edge TTS & Pygame

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/rohitvaddepalli/Final-Year-Project.git
cd "Final Project"
```

### 2. Set up a Virtual Environment
```powershell
python -m venv env
.\env\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install System Dependency (FFmpeg)
Whisper requires **FFmpeg** to process audio recordings.
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your System PATH.
- **Verification:** Run `ffmpeg -version` in your terminal to ensure it's accessible.

## 📖 Usage
Run the main interface to start the application:
```powershell
python interface_memory.py
```
1. Click **"Start Recording"**.
2. Speak naturally (the system stops automatically when you finish).
3. View your transcribed text, detected emotion, and generated motivation.
4. Listen to the voice response and check the **History** or **Statistics** tabs for analysis.

## 📊 Project Structure
- `interface_memory.py`: The main GUI and memory management system.
- `motivation_generator.py`: Module for generating empathetic responses.
- `requirements.txt`: List of Python dependencies.
- `conversation_history.json`: Local storage for interaction memory (persisted).

## 📄 Documentation
A detailed Survey Paper is included in the project root:
- `Survey paper_demo.pdf`: Comprehensive research and methodology documentation.

## ⚖️ License
This project is developed as part of a 4th Year Final Project.
