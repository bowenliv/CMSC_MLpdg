from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, concatenate_datasets
import torch
from torch import nn
import numpy as np
import evaluate
# Load the dataset
dataset = load_dataset("multi_nli")
# Filter the training set to include only the genre you want to train on
source_dataset = dataset["train"].filter(lambda example: example["genre"] == "travel")
#val_source_dataset = dataset["validation_matched"].filter(lambda example: example["genre"] == "travel")

telephone = dataset["train"].filter(lambda example: example["genre"] == "telephone")
target_dataset = telephone.select(range(8335))
val_target_dataset = dataset["validation_matched"].filter(lambda example: example["genre"] == "telephone")

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
metric = evaluate.load("accuracy")

# Tokenize the training dataset
def tokenize_dataset(dataset):
    return dataset.map(lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128), batched=True)
train_source_dataset = tokenize_dataset(source_dataset)
eval_source_dataset = tokenize_dataset(val_target_dataset)

model_domain = AutoModelForSequenceClassification.from_pretrained("./results_text_classifier/checkpoint-3000", num_labels=2)
model_domain = model_domain.cuda()
def calculate_weight(data):
    new_data = {}
    new_data['input_ids'] = torch.tensor(data['input_ids']).unsqueeze(0).cuda()
    new_data['token_type_ids'] = torch.tensor(data['token_type_ids']).unsqueeze(0).cuda()
    new_data['attention_mask'] = torch.tensor(data['attention_mask']).unsqueeze(0).cuda()
    outputs = model_domain(**new_data)
    #print(outputs)
    percentage = torch.nn.functional.softmax(outputs.get("logits"), dim=-1)
    percentage = percentage.reshape(-1).tolist()
    data['label'] = [data['label'], (percentage[1] * 0.885) / (percentage[0] * 0.115)]
    return data

print('calculating weight...')
train_source_dataset = train_source_dataset.map(calculate_weight)
eval_source_dataset = eval_source_dataset.map(calculate_weight)
print('weight fished!')

def compute_metrics_source(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels[:, :1])

def compute_metrics_target(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_all = inputs.get("labels")
        labels = labels_all[:, :1]
        instance_weight = labels_all[:, 1:]
        # forward pass
        outputs = model(input_ids=inputs.get("input_ids"), token_type_ids=inputs.get("token_type_ids"), attention_mask=inputs.get("attention_mask"))
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.long().view(-1))
        loss = loss*instance_weight
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results_reweighting_source_new",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs_reweighting_source_new",
    logging_steps=len(train_source_dataset)//128,
    save_steps=len(train_source_dataset)//128,
)
# Define the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_source_dataset,
    eval_dataset=eval_source_dataset,
    compute_metrics=compute_metrics_source,
)
# Train the model
trainer.train()



train_target_dataset = tokenize_dataset(target_dataset)
eval_target_dataset = tokenize_dataset(val_target_dataset)
# Define the training arguments
training_args_target = TrainingArguments(
    output_dir="./results_reweighting_target_new",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs_reweighting_target_new",
    logging_steps=len(train_target_dataset)//128,
    save_steps=len(train_target_dataset)//128,
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
trainer_target.train()


