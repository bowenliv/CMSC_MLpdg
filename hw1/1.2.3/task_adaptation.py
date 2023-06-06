import torch
from transformers import pipeline, AutoTokenizer, DefaultDataCollator, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, Trainer
import datasets
import evaluate
import numpy as np

EXPERIMENT = "bert-squad-10"
MODEL_PATH = "/fs/classhomes/spring2023/cmsc828a/c828a007/c828ag02/saved_models/" + EXPERIMENT
LOG_PATH = "/fs/classhomes/spring2023/cmsc828a/c828a007/c828ag02/logs/" + EXPERIMENT

"""# Pretrain BERT on SQuADv2"""

squad = datasets.load_dataset("squad_v2")
#squad['train'] = squad['train'].select(range(65160))
#squad['train'] = squad['train'].select(range(26064))
#squad['train'] = squad['train'].select(range(13032))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

pad_on_right = tokenizer.padding_side == "right"
max_length = 384
doc_stride = 128

def preprocess_function(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")

training_args = TrainingArguments(
    output_dir=LOG_PATH + "-squad",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=LOG_PATH + "-squad",
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

"""# Load MNLI dataset
Select only "telephone" examples as the target dataset, and extract a 10% split to use for fine-tuning. 
"""

mnli_dataset = datasets.load_dataset("multi_nli")
telephone_dataset = mnli_dataset.filter(lambda example: example["genre"] == "telephone")
# There are 83348 examples in the original MNLI training data in the “telephone” genre, so selecting the first 8335 examples is 10%. 
telephone_dataset['train'] = telephone_dataset['train'].select(range(8335))
del telephone_dataset['validation_mismatched']

"""# Tokenize dataset

Using bert-base-cased-squad2 tokenizer
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding='max_length', max_length=128)
tokenized_dataset = telephone_dataset.map(tokenize_function, batched=True)
#tokenized_dataset = tokenized_dataset.select_columns(['label', 'input_ids', 'token_type_ids', 'attention_mask'])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format(type="torch", columns=["labels", "input_ids", "token_type_ids", "attention_mask"])
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

"""# Load Model and define Trainer

Using a pretrained model: BERT base cased model trained on SQuAD v2. 

Trainer parameters:
  - Learning rate (default): 2e-5
  - per_device_train_batch_size=16
  - per_device_eval_batch_size=64
  - weight_decay (default): 0.01
  - num_train_epochs = 5
"""

#mnli_model = AutoModelForSequenceClassification.from_pretrained("deepset/bert-base-cased-squad2", num_labels=3)
mnli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True, num_labels=3)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=LOG_PATH + "-mnli",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=LOG_PATH + "-mnli",
    logging_steps=500,
)

trainer = Trainer(
    mnli_model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation_matched"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

"""# Train the model"""

trainer.train()

"""# Calculate LEEP Score"""

def LEEP(predictions: np.ndarray, labels: np.ndarray):
    """
    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    N, C_s = predictions.shape
    labels = labels.reshape(-1)
    C_t = int(np.max(labels) + 1)

    normalized_prob = predictions / float(N)
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)

    for i in range(C_t):
        this_class = normalized_prob[labels == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = predictions @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, labels)])
    score = np.mean(np.log(empirical_prob))

    return score

output = trainer.predict(tokenized_dataset["validation_matched"])
leep_score = LEEP(torch.softmax(torch.Tensor(output.predictions), dim=1).numpy(), output.label_ids)
print("LEEP SCORE: {}".format(leep_score))

