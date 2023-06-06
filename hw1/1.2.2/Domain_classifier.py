from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, concatenate_datasets
import torch
import numpy as np
import evaluate
# Load the dataset
dataset = load_dataset("multi_nli")
# Filter the training set to include only the genre you want to train on
source_dataset = dataset["train"].filter(lambda example: example["genre"] == "travel")
labeled_source_dataset = source_dataset.map(lambda example: {"label": 0})
telephone = dataset["train"].filter(lambda example: example["genre"] == "telephone")
target_dataset = telephone.select(range(8335))
labeled_target_dataset = target_dataset.map(lambda example: {"label": 1})
train_dataset = concatenate_datasets([labeled_source_dataset, labeled_target_dataset])

val_source_dataset = dataset["validation_matched"].filter(lambda example: example["genre"] == "travel")
labeled_val_source_dataset = val_source_dataset.map(lambda example: {"label": 0})
val_target_dataset = dataset["validation_matched"].filter(lambda example: example["genre"] == "telephone")
labeled_val_target_dataset = val_target_dataset.map(lambda example: {"label": 1})
val_dataset = concatenate_datasets([labeled_val_source_dataset, labeled_val_target_dataset])

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
metric = evaluate.load("accuracy")

# Tokenize the training dataset
def tokenize_dataset(dataset):
    return dataset.map(lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128), batched=True)

train_dataset = tokenize_dataset(train_dataset)
eval_dataset = tokenize_dataset(val_dataset)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results_text_classifier",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)
# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
# Train the model
trainer.train()
