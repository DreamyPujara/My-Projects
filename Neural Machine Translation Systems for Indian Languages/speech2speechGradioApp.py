 import gradio as gr
    import speech_recognition as sr
    from transformers import MarianMTModel, MarianTokenizer
    from gtts import gTTS
    import os

    # Load models and tokenizers
    model_en_hi = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
    tokenizer_en_hi = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

    def speech_to_speech_translation(audio):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        translated = translate_text(text, model_en_hi, tokenizer_en_hi)

        tts = gTTS(translated, lang='hi')
        tts.save('output.mp3')

        return 'output.mp3'

    def translate_text(text, model, tokenizer):
        inputs = tokenizer.encode(text, return_tensors='pt')
        translated = model.generate(inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    gr.Interface(fn=speech_to_speech_translation, inputs=gr.Audio(source="microphone"), outputs=gr.Audio()).launch()
