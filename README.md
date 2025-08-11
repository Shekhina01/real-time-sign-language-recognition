# Real-Time Recognition of Sign Language

A deep learning-based system that recognises American and Indian Sign Language gestures in real-time using a Convolutional Neural Network (CNN).  
The system translates gestures into text and converts them into audible speech using the Google Speech API.  
It also supports reverse translation — converting speech into corresponding sign language animations.

## Features
- **Sign to Audio**: Captures hand gestures via webcam, recognises them using a trained CNN, and converts to speech.
- **Audio to Sign**: Converts live or recorded speech into sign language animations.
- **Real-time Processing**: Achieves recognition with minimal latency using OpenCV and TensorFlow.
- **Multi-language Support**: Currently supports ASL & ISL alphabets and words.

## Tech Stack
- **Languages**: Python 3.7+
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, pyttsx3, speechrecognition, BeautifulSoup
- **Tools**: Anaconda, Jupyter Notebook
- **Hardware**: Standard webcam

## Installation
```bash
git clone https://github.com/Shekhina01/real-time-sign-language-recognition.git
cd real-time-sign-language-recognition
pip install -r requirements.txt

# Speech → Sign
python src/audio_to_sign.py

# Webcam Sign → Text (and optional speech)
python src/sign_to_audio.py
