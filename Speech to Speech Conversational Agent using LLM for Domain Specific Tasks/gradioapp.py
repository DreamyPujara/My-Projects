import gradio as gr

def speech_to_speech(audio):
    # Step 1: Convert speech to text
    text = speech_to_text(audio)
    
    # Step 2: Translate text to English
    english_text = translate_to_english(text, src_lang="hi")
    
    # Step 3: Generate response using LLM
    response = generate_response(english_text)
    
    # Step 4: Convert response to speech
    response_audio_path = text_to_speech(response)
    
    return response_audio_path

iface = gr.Interface(fn=speech_to_speech, inputs=gr.Audio(source="microphone"), outputs=gr.Audio())
iface.launch()
