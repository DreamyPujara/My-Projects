from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load LLM model and tokenizer
llm_model = GPT2LMHeadModel.from_pretrained("gpt2")
llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(text):
    inputs = llm_tokenizer.encode(text, return_tensors="pt")
    outputs = llm_model.generate(inputs, max_length=100, num_return_sequences=1)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
# response = generate_response(english_text)
