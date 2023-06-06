from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, concatenate_datasets, DownloadConfig
import torch
import numpy as np 
import evaluate 

task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "bert-base-cased"
batch_size = 16


#dd = DownloadConfig(cache_dir="wikineural_data/")

datasets = {}
datasets['train'] = load_dataset("Babelscape/wikineural", split="train_en").shuffle(seed=42)
datasets['val'] = load_dataset("Babelscape/wikineural", split="val_en").shuffle(seed=42)
datasets['test'] = load_dataset("Babelscape/wikineural", split="test_en").shuffle(seed=42)

datasets["train"].features[f"ner_tags"]

tag_set = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
tag_set = {v: k for k, v in tag_set.items()}

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


for key in datasets:
    datasets[key] = datasets[key].map(lambda example: {'ner_tags_named': [tag_set[tag] for tag in example['ner_tags']]})
    print(key, ':', datasets[key])



def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

label_all_tokens = True
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_datasets = {}
for key in datasets:
    tokenized_datasets[key] = datasets[key].map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True)


# import pdb;pdb.set_trace()

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{task}_wikineural",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False, #True,
    save_strategy = "epoch",
    logging_strategy = "epoch",
)


data_collator = DataCollatorForTokenClassification(tokenizer)


metric = load_metric("seqeval")

# Define the trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
