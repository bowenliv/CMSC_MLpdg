from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import torch
import numpy as np 
import evaluate 

model_checkpoint = "bert-base-cased"

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3, ignore_mismatched_sizes=True)
metric = evaluate.load("accuracy")

mnli_dataset = load_dataset("multi_nli")
source_train = mnli_dataset["train"]
source_val = mnli_dataset["validation_matched"]

target_train = mnli_dataset['train'].filter(lambda example: example["genre"] == "telephone")
target_train = target_train.select(range(8335))
target_val = mnli_dataset["validation_matched"].filter(lambda example: example["genre"] == "telephone")


# Tokenize the training dataset
def tokenize_dataset(dataset):
    return dataset.map(lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128), batched=True)
train_dataset = tokenize_dataset(source_train)
eval_dataset = tokenize_dataset(source_val)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# import pdb;pdb.set_trace()

model_name = model_checkpoint.split("/")[-1]
# Define the training arguments
training_args = TrainingArguments(
    f"{model_name}-finetuned-NLI_mnli",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=f"logs_{model_name}_NLI/",
    # logging_steps=10000,
    logging_strategy = "epoch",
    save_strategy = "epoch",
    # load_best_model_at_end = True,
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
