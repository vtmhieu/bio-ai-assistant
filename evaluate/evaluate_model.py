import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "./model_output/best_model"
LABEL_NAMES = ["background", "conclusions", "methods", "objective", "results"]
label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

# Load the model as a pipeline
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    device="mps",
    top_k=None,
)

# Load test data
dataset = load_dataset("armanc/pubmed-rct20k")
test_data = dataset["test"]

# Map string labels to integers (same mapping as training)
true_labels = [label2id[label] for label in test_data["label"]]
texts = test_data["text"]

print(f"Evaluating on {len(test_data)} test samples...")

# Batch prediction
predictions = []
batch_size = 64
for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    results = classifier(batch, truncation=True, max_length=128)
    for result in results:
        best = max(result, key=lambda x: x["score"])
        pred_idx = label2id[best["label"]]
        predictions.append(pred_idx)
    if i % 500 == 0:
        print(f"  Processed {i}/{len(texts)}...")

# --- Classification report ---
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(true_labels, predictions, target_names=LABEL_NAMES))

# --- Confusion matrix ---
print("\nCONFUSION MATRIX")
cm = confusion_matrix(true_labels, predictions)
print(f"{'':>14}", end="")
for name in LABEL_NAMES:
    print(f"{name[:10]:>12}", end="")
print()
for i, row in enumerate(cm):
    print(f"{LABEL_NAMES[i]:>14}", end="")
    for val in row:
        print(f"{val:>12}", end="")
    print()

# --- Error analysis ---
print("\n\nWORST PREDICTIONS (highest-confidence mistakes):")
count = 0
for i in range(len(predictions)):
    if predictions[i] != true_labels[i] and count < 5:
        print(f"  Text: {texts[i][:100]}...")
        print(
            f"  True: {LABEL_NAMES[true_labels[i]]}, Predicted: {LABEL_NAMES[predictions[i]]}"
        )
        print()
        count += 1
