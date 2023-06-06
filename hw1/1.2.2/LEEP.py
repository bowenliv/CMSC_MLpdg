from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, concatenate_datasets
import torch
from torch import nn
import numpy as np
import evaluate

import numpy as np

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
# Filter the training set to include only the genre you want to train on

telephone = dataset["train"].filter(lambda example: example["genre"] == "telephone")
target_dataset = telephone.select(range(8335))
val_target_dataset = dataset["validation_matched"].filter(lambda example: example["genre"] == "telephone")

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("./results_reweighting_target_new/checkpoint-325", num_labels=3)
metric = evaluate.load("accuracy")

# Tokenize the training dataset
def tokenize_dataset(dataset):
    return dataset.map(lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128), batched=True)


def compute_metrics_target(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


train_target_dataset = tokenize_dataset(target_dataset)
eval_target_dataset = tokenize_dataset(val_target_dataset)
# Define the training arguments
training_args_target = TrainingArguments(
    output_dir="./results_reweighting_target_v2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs_reweighting_target_v2",
    logging_steps=len(train_target_dataset)//128,
)
# Define the trainer
trainer_target = Trainer(
    model=model,
    args=training_args_target,
    train_dataset=train_target_dataset,
    eval_dataset=eval_target_dataset,
    compute_metrics=compute_metrics_target,
)
# Train the model

output = trainer_target.predict(eval_target_dataset)
leep_score = LEEP(torch.softmax(torch.Tensor(output.predictions), dim=1).numpy(), output.label_ids)

print(leep_score)