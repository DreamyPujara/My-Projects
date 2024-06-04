from transformers import MarianMTModel, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets

# Load preprocessed data
general_df = pd.read_csv('preprocessed_general_domain.csv')
specific_df = pd.read_csv('preprocessed_specific_domain.csv')

# Create datasets
general_dataset = Dataset.from_pandas(general_df)
specific_dataset = Dataset.from_pandas(specific_df)

# Initialize model
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_curriculum',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs_curriculum',
)

# Curriculum Learning: Train first on general domain, then on specific domain
# Train on general domain first
trainer_general = Trainer(
    model=model,
    args=training_args,
    train_dataset=general_dataset,
)

trainer_general.train()

# Train on specific domain next
trainer_specific = Trainer(
    model=model,
    args=training_args,
    train_dataset=specific_dataset,
)

trainer_specific.train()
