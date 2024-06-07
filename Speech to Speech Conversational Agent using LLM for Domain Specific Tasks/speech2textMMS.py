import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load MMS model and processor
mms_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
mms_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def speech_to_text(audio_path):
    # Load audio
    speech, rate = torchaudio.load(audio_path)
    speech = speech.squeeze()

    # Resample if necessary
    if rate != 16000:
        speech = torchaudio.transforms.Resample(rate, 16000)(speech)

    # Tokenize and infer
    input_values = mms_processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    logits = mms_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = mms_processor.decode(predicted_ids[0])
    return transcription

# Example usage
# text = speech_to_text("path_to_audio.wav")
