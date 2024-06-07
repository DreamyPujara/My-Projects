from gtts import gTTS
import os

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("response.mp3")
    return "response.mp3"

# Example usage
# audio_path = text_to_speech(response)
