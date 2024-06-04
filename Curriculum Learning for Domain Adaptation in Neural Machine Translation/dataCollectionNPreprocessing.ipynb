import os
import pandas as pd
from transformers import MarianTokenizer

# Example dataset paths
general_domain_path = 'data/general_domain.csv'
specific_domain_path = 'data/specific_domain.csv'

# Load datasets
general_df = pd.read_csv(general_domain_path)
specific_df = pd.read_csv(specific_domain_path)

# Initialize tokenizer
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Tokenize and preprocess data
def preprocess(text):
    return tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True)

general_df['input_ids'] = general_df['source'].apply(preprocess)
general_df['labels'] = general_df['target'].apply(preprocess)

specific_df['input_ids'] = specific_df['source'].apply(preprocess)
specific_df['labels'] = specific_df['target'].apply(preprocess)

# Save preprocessed data
general_df.to_csv('preprocessed_general_domain.csv', index=False)
specific_df.to_csv('preprocessed_specific_domain.csv', index=False)
