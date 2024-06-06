  from transformers import MarianMTModel, MarianTokenizer

    def train_nmt_model(src_lang, tgt_lang, train_data):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Training code here using the `train_data`
        # Example: Fine-tune using Trainer API from Hugging Face

        return model, tokenizer
