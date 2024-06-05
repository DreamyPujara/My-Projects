import pandas as pd
from transformers import MarianTokenizer

# Load the dataset
news_df = pd.read_csv('news_data.csv')

# Normalize and tokenize
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

def preprocess(text):
    return tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True)

news_df['content_tokenized'] = news_df['content'].apply(preprocess)
news_df.to_csv('preprocessed_news_data.csv', index=False)
