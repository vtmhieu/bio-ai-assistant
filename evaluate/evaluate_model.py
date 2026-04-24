import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "./model_output/best_model"
LABEL_NAMES = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]

# Load the model as a pipeline (same pattern you'll use in production)
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    device="mps",  # or "cpu"
    top_k=None,  # return all class scores
)

# Load test data
dataset = load_dataset("qanastek/pubmed-rct20k")
test_data = dataset["test"]

# Predict in batches
print(f"Evaluating on {len(test_data)} test samples...")
texts = test_data["text"]
true_labels = test_data["label"]

# Batch prediction
predictions = []
batch_size = 64
for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    results = classifier(batch, truncation=True, max_length=128)
    for result in results:
        # Get the label with the highest score
        best = max(result, key=lambda x: x["score"])
        pred_idx = int(best["label"].split("_")[-1])
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
    print(f"{name[:8]:>10}", end="")
print()
for i, row in enumerate(cm):
    print(f"{LABEL_NAMES[i][:14]:>14}", end="")
    for val in row:
        print(f"{val:>10}", end="")
    print()

# --- Error analysis: find worst predictions ---
print("\n\nWORST PREDICTIONS (highest-confidence mistakes):")
for i in range(min(5, len(predictions))):
    if predictions[i] != true_labels[i]:
        print(f"  Text: {texts[i][:100]}...")
        print(
            f"  True: {LABEL_NAMES[true_labels[i]]}, Predicted: {LABEL_NAMES[predictions[i]]}"
        )
        print()
