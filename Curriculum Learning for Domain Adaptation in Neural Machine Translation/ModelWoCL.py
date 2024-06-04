from transformers import MarianMTModel, Trainer, TrainingArguments
from datasets import Dataset

# Load preprocessed data
general_df = pd.read_csv('preprocessed_general_domain.csv')
specific_df = pd.read_csv('preprocessed_specific_domain.csv')

# Create datasets
general_dataset = Dataset.from_pandas(general_df)
specific_dataset = Dataset.from_pandas(specific_df)

# Combine datasets for training
combined_dataset = Dataset.concatenate([general_dataset, specific_dataset])

# Initialize model
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_no_curriculum',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs_no_curriculum',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
)

# Train the model
trainer.train()
