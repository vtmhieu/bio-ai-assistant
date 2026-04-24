import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# --- Config ---
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
OUTPUT_DIR = "./model_output"
EPOCHS = 2
BATCH_SIZE = 32

# --- Load dataset ---
print("Loading PubMed RCT dataset...")
dataset = load_dataset("armanc/pubmed-rct20k")

# Explore the data — understand what you're working with
print(f"Train size: {len(dataset['train'])}")
print(f"Labels: {dataset['train'].features['label']}")
print(f"Example: {dataset['train'][0]}")

# --- Map string labels to integers ---
label_names = ["background", "conclusions", "methods", "objective", "results"]
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for i, name in enumerate(label_names)}


def map_labels(example):
    example["label"] = label2id[example["label"]]
    return example


print("Mapping labels...")
dataset = dataset.map(map_labels)

# --- Tokenize ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


print("Tokenizing...")
tokenized = dataset.map(tokenize, batched=True, batch_size=1000)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# --- Define metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# --- Load model ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# --- Train ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

print("Training started...")
trainer.train()

# --- Save ---
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
print(f"Model saved to {OUTPUT_DIR}/best_model")
