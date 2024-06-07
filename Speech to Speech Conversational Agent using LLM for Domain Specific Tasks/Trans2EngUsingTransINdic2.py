from transformers import MarianMTModel, MarianTokenizer

# Load IndicTrans2 model and tokenizer
model_name = "AI4Bharat/indic-bert"
indictrans_model = MarianMTModel.from_pretrained(model_name)
indictrans_tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_to_english(text, src_lang="hi"):
    inputs = indictrans_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = indictrans_model.generate(**inputs)
    translated_text = indictrans_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Example usage
# english_text = translate_to_english(text, src_lang="hi")
