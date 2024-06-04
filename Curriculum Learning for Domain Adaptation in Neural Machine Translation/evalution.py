from transformers import MarianTokenizer, MarianMTModel
from datasets import load_metric

# Load test dataset
test_df = pd.read_csv('preprocessed_test_data.csv')
test_dataset = Dataset.from_pandas(test_df)

# Load models
model_no_curriculum = MarianMTModel.from_pretrained('./results_no_curriculum')
model_curriculum = MarianMTModel.from_pretrained('./results_curriculum')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')

# Function to compute metrics
def compute_metrics(predictions, references):
    metric = load_metric('sacrebleu')
    predictions = [pred.strip() for pred in predictions]
    references = [[ref.strip()] for ref in references]
    return metric.compute(predictions=predictions, references=references)

# Evaluate model without curriculum learning
inputs = tokenizer(test_df['source'].tolist(), return_tensors='pt', padding=True, truncation=True)
outputs = model_no_curriculum.generate(**inputs)
predictions_no_curriculum = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
metrics_no_curriculum = compute_metrics(predictions_no_curriculum, test_df['target'].tolist())

# Evaluate model with curriculum learning
outputs = model_curriculum.generate(**inputs)
predictions_curriculum = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
metrics_curriculum = compute_metrics(predictions_curriculum, test_df['target'].tolist())

print("Metrics without Curriculum Learning:", metrics_no_curriculum)
print("Metrics with Curriculum Learning:", metrics_curriculum)
