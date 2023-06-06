from transformers import AutoTokenizer, BertModel, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, concatenate_datasets
from torch import nn
import torch
import numpy as np
import evaluate
from timm.models.layers import trunc_normal_, DropPath

domain_class_num=2
classifier_class_num=3


class DANN(nn.Module):

    def __init__(self, domain_class_num=2, classifier_class_num=3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.domain_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, domain_class_num),
            nn.GELU(),
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, classifier_class_num),
            nn.GELU(),
        )
        for m in self.domain_head:
            self._init_weights(m)
        for m in self.classifier_head:
            self._init_weights(m)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs.get("pooler_output")
        domain = self.domain_head(logits)
        classifier = self.classifier_head(logits)
        outputs['domain'] = domain
        outputs['classifier'] = classifier
        outputs['logits'] = classifier
        #outputs['labels'] = labels

        return outputs

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


dataset = load_dataset("multi_nli")
# Filter the training set to include only the genre you want to train on
source_dataset = dataset["train"].filter(lambda example: example["genre"] == "travel")
labeled_source_dataset = source_dataset.map(lambda example: {"label": [0, example["label"]]})
telephone = dataset["train"].filter(lambda example: example["genre"] == "telephone")
target_dataset = telephone.select(range(8335))
labeled_target_dataset = target_dataset.map(lambda example: {"label": [1, example["label"]]})
train_dataset = concatenate_datasets([labeled_source_dataset, labeled_target_dataset])

val_target_dataset = dataset["validation_matched"].filter(lambda example: example["genre"] == "telephone")
labeled_val_target_dataset = val_target_dataset.map(lambda example: {"label": [1, example["label"]]})
val_dataset = labeled_val_target_dataset

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = DANN(domain_class_num=domain_class_num, classifier_class_num=classifier_class_num)
metric = evaluate.load("accuracy")
# Tokenize the training dataset
def tokenize_dataset(dataset):
    return dataset.map(lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128), batched=True)

train_dataset = tokenize_dataset(train_dataset)
eval_dataset = tokenize_dataset(val_dataset)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # print(logits)
    # print('******************')
    # print(labels)
    # print(logits.size, labels.size)
    predictions = np.argmax(logits, axis=-1)
    #exit(0)
    return metric.compute(predictions=predictions, references=labels[:, 1:])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_all = inputs.get("labels")
        domain_labels = labels_all[:, :1]
        classifier_labels = labels_all[:, 1:]
        # forward pass
        outputs = model(input_ids=inputs.get("input_ids"), token_type_ids=inputs.get("token_type_ids"), attention_mask=inputs.get("attention_mask"), labels=classifier_labels)
        domain_logits = outputs.get("domain")
        classifier_logits = outputs.get("classifier")

        loss_fct_domain = nn.CrossEntropyLoss()
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        domain_loss = loss_fct_domain(domain_logits.view(-1, domain_class_num), domain_labels.long().view(-1))

        classifier_loss = loss_fct(classifier_logits.view(-1, classifier_class_num), classifier_labels.long().view(-1))

        percentage = torch.nn.functional.softmax(domain_logits, dim=-1)
        instance_weight = (percentage[:,1:] * 0.885) / (percentage[:, :1] * 0.115)

        classifier_loss = (classifier_loss*instance_weight).mean()
        loss = domain_loss + classifier_loss
        return (loss, outputs) if return_outputs else loss

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results_DANN_v2",
    evaluation_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs_DANN_v2",
    logging_steps=len(train_dataset)//128,
    save_steps=len(train_dataset)//128,
)
# Define the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
# Train the model
trainer.train(ignore_keys_for_eval=['last_hidden_state', 'pooler_output', 'domain', 'classifier', 'labels'])

