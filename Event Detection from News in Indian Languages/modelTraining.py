import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the preprocessed data
news_df = pd.read_csv('preprocessed_news_data.csv')

# Create a dataset
dataset = Dataset.from_pandas(news_df)

# Split into train and test sets
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()

# Load a pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
