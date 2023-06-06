# -*- coding: utf-8 -*-
"""Domain_adaptation_baseline2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ytr8cdi6xVOty7ybrndkhhPPKMsLrVZb
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import torch
import numpy as np 
import evaluate 

# leep score
def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    """

    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)

    empirical_prediction = pseudo_source_label @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    return leep_score

# Load the dataset
dataset = load_dataset("multi_nli")

train_dataset = dataset["train"].filter(lambda example: example["genre"] == "telephone")
# Select the first 10% of the filtered training set
train_dataset = train_dataset.select(range(8335))
dev_data = dataset['validation_matched'].filter(lambda example: example['genre'] == 'telephone')


# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# load the model checkpoint from baseline 1 model
model_path = "./results/checkpoint-12000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

metric = evaluate.load("accuracy")

# Tokenize the training dataset
def tokenize_dataset(dataset):
    return dataset.map(lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128), batched=True)

train_dataset = tokenize_dataset(train_dataset)
eval_dataset = tokenize_dataset(dev_data)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="steps",  # save checkpoints every "save_steps"
    save_steps=4000,  # save a checkpoint every 2000 steps
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


# leep score calculation
eval_dataset = eval_dataset.rename_column("label", "labels")
eval_dataset.set_format(type="torch", columns=["labels", "input_ids", "token_type_ids", "attention_mask"])
output = trainer.predict(eval_dataset)
leep_score = LEEP(torch.softmax(torch.Tensor(output.predictions), dim=1).numpy(), output.label_ids)
print("leep_score", leep_score)
