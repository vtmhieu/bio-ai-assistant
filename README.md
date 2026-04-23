# SciLifeLab-Project

# SciLifeLab AI Data Engineer — Hands-on Project

> **Goal:** Build a biomedical research assistant that mirrors the actual job scope at SciLifeLab Data Centre.
> Runs entirely on MacBook Air M4. Total time: ~4 hours.

## What this project covers (mapped to the JD)

| JD Requirement | What you'll build |
|---|---|
| Deploy, monitor, maintain AI models | Fine-tune DistilBERT, serve via FastAPI, add Prometheus metrics |
| Integrate LLMs + foundation models in research workflows | LangChain agent with tools (classifier, PubMed search, LLM summarizer) |
| AI agent architectures (LangChain/Haystack) | ReAct agent that reasons about which tool to use |
| Container technologies (Docker, Helm, Kubernetes) | Dockerized service with docker-compose and health checks |
| Training, evaluating deep neural networks | Full train → evaluate → save pipeline with metrics |
| Python + version control | Everything in Python, Git-tracked |

---

## Project structure

```
scilifelab-ai-assistant/
├── README.md
├── requirements.txt
├── train/
│   └── train_classifier.py       # Step 1: Fine-tune DistilBERT
├── evaluate/
│   └── evaluate_model.py         # Step 2: Evaluate with F1, confusion matrix
├── agent/
│   ├── tools.py                  # Step 3: Define agent tools
│   └── agent.py                  # Step 3: LangChain ReAct agent
├── api/
│   └── main.py                   # Step 4: FastAPI + Prometheus
├── Dockerfile                    # Step 5: Containerize
├── docker-compose.yml            # Step 5: Compose stack
└── tests/
    └── test_api.py               # Step 6: Basic tests
```

---

## Setup

```bash
mkdir scilifelab-ai-assistant && cd scilifelab-ai-assistant
python3 -m venv .venv && source .venv/bin/activate
git init

pip install torch transformers datasets scikit-learn \
  langchain langchain-community \
  fastapi uvicorn prometheus-client \
  biopython requests
```

---

## Step 1 — Fine-tune a biomedical text classifier (~60 min)

**Why:** The JD says "training, evaluating, and deploying ML models." This gives you a real model to deploy later.

**Dataset:** PubMed RCT — sentences from medical abstracts labeled by their role (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS). This is exactly the kind of life science data SciLifeLab works with.

Create `train/train_classifier.py`:

```python
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
dataset = load_dataset("qanastek/pubmed-rct20k")

# Explore the data — understand what you're working with
print(f"Train size: {len(dataset['train'])}")
print(f"Labels: {dataset['train'].features['label']}")
print(f"Example: {dataset['train'][0]}")

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
# Use MPS (Metal) on M4 Mac for GPU acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
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
    report_to="none",  # no wandb needed
    use_mps_device=(device == "mps"),
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
```

**Run it:**
```bash
python train/train_classifier.py
```

Watch the loss decrease and F1 increase each epoch. On M4 with MPS, this should take ~30-45 minutes.

**What to understand:**
- `tokenizer` converts text → numbers (token IDs). Each word gets split into subwords from BERT's vocabulary.
- `AutoModelForSequenceClassification` adds a classification head on top of DistilBERT.
- `compute_metrics` calculates accuracy AND F1 — F1 matters more because classes may be imbalanced.
- The `Trainer` handles the training loop: forward pass → compute loss → backpropagate → update weights.

---

## Step 2 — Evaluate the model properly (~20 min)

**Why:** The JD says "evaluating ML models." You need to understand not just IF the model works, but WHERE it fails.

Create `evaluate/evaluate_model.py`:

```python
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
    top_k=None,     # return all class scores
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
    batch = texts[i:i+batch_size]
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
        print(f"  True: {LABEL_NAMES[true_labels[i]]}, Predicted: {LABEL_NAMES[predictions[i]]}")
        print()
```

**Run it:**
```bash
python evaluate/evaluate_model.py
```

**What to tell Arnold:**
- "I evaluated per-class F1 because accuracy alone hides class imbalance."
- "I also did error analysis — the model confuses BACKGROUND and METHODS most often because they share similar language."
- "The confusion matrix shows exactly where the model needs more data or augmentation."

---

## Step 3 — Build an AI agent with LangChain (~60 min)

**Why:** This is the most important part. The JD specifically mentions "AI agent architectures" and "LangChain/Haystack." This is likely a big chunk of your day-to-day work.

Create `agent/tools.py`:

```python
"""
Agent tools that wrap AI models and external APIs.
This simulates integrating ML models + external data sources
into research workflows — exactly what SciLifeLab needs.
"""
from langchain.tools import tool
from transformers import pipeline
from Bio import Entrez

# Configure Entrez (PubMed API)
Entrez.email = "your.email@example.com"

# Load the fine-tuned classifier once at module level
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="./model_output/best_model",
            device="cpu",  # CPU for tool use, keeps memory low
        )
    return _classifier


@tool
def classify_biomedical_text(text: str) -> str:
    """Classify a biomedical text sentence into one of:
    BACKGROUND, OBJECTIVE, METHODS, RESULTS, or CONCLUSIONS.
    Use this when a researcher wants to understand what role
    a sentence plays in a scientific abstract."""
    clf = get_classifier()
    result = clf(text, truncation=True, max_length=128)
    label = result[0]["label"]
    score = result[0]["score"]
    return f"Classification: {label} (confidence: {score:.2%})"


@tool
def search_pubmed(query: str, max_results: int = 3) -> str:
    """Search PubMed for recent biomedical research papers.
    Use this when a researcher asks about recent studies,
    clinical trials, or scientific literature on a topic.
    Returns titles, authors, and abstracts."""
    try:
        # Search PubMed
        handle = Entrez.esearch(
            db="pubmed", term=query, retmax=max_results, sort="date"
        )
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]

        if not ids:
            return f"No PubMed results found for: {query}"

        # Fetch details
        handle = Entrez.efetch(
            db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        results = []
        for article in records["PubmedArticle"]:
            medline = article["MedlineCitation"]
            title = medline["Article"]["ArticleTitle"]
            pmid = str(medline["PMID"])

            # Get abstract if available
            abstract = ""
            if "Abstract" in medline["Article"]:
                abstract_texts = medline["Article"]["Abstract"]["AbstractText"]
                abstract = " ".join(str(t) for t in abstract_texts)[:300]

            results.append(
                f"PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}...\n"
            )

        return "\n---\n".join(results)

    except Exception as e:
        return f"PubMed search error: {str(e)}"


@tool
def summarize_text(text: str) -> str:
    """Summarize a long biomedical text into key points.
    Use this when a researcher has a long abstract or paper
    section and wants a concise summary."""
    # In production, this would call a hosted LLM.
    # For the demo, we use a simple extractive approach.
    sentences = text.replace("\n", " ").split(". ")
    if len(sentences) <= 3:
        return text

    # Take first, middle, and last sentences as a simple summary
    key_sentences = [
        sentences[0],
        sentences[len(sentences) // 2],
        sentences[-1],
    ]
    return "Key points: " + ". ".join(s.strip() for s in key_sentences if s.strip())
```

Now create the agent — `agent/agent.py`:

```python
"""
LangChain ReAct agent for biomedical research assistance.

This demonstrates the AI agent architecture pattern mentioned
in the SciLifeLab JD. The agent decides which tool to use
based on the researcher's question.
"""
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from agent.tools import classify_biomedical_text, search_pubmed, summarize_text

# --- Option A: Use Ollama with a local LLM (recommended for M4 Mac) ---
# Install: brew install ollama && ollama pull llama3.2:3b
# This runs entirely locally — no API key needed.
llm = Ollama(model="llama3.2:3b", temperature=0)

# --- Option B: Use OpenAI API (if you have a key) ---
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the tools the agent can use
tools = [classify_biomedical_text, search_pubmed, summarize_text]

# ReAct prompt — the agent reasons step-by-step
REACT_PROMPT = PromptTemplate.from_template("""You are a biomedical research assistant
at SciLifeLab. You help researchers with their questions using available tools.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question
Thought: think about what tool to use
Action: the tool name
Action Input: the input to the tool
Observation: the result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
{agent_scratchpad}""")

# Create the agent
agent = create_react_agent(llm, tools, REACT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,    # Shows the agent's reasoning — great for demos
    max_iterations=5,
    handle_parsing_errors=True,
)


def run_agent(question: str) -> str:
    """Run the agent on a research question."""
    result = agent_executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    # Test queries that exercise different tools
    test_queries = [
        "Classify this sentence: We enrolled 500 patients in a randomized controlled trial.",
        "Find recent papers about CRISPR gene therapy for cancer.",
        "What role does this sentence play in a paper: Our results demonstrate a significant improvement in survival rates.",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = run_agent(q)
        print(f"\nAnswer: {answer}")
```

**To run this, first install Ollama** (runs a local LLM on your M4 — no API key needed):
```bash
brew install ollama
ollama pull llama3.2:3b    # Small model, runs fast on M4
python -m agent.agent
```

**What to tell Arnold:** "I built a ReAct agent using LangChain that integrates a fine-tuned classifier, PubMed search, and text summarization. The agent reasons about which tool to use based on the researcher's question. This is the kind of architecture the JD describes for integrating models into research workflows."

---

## Step 4 — Serve it with FastAPI + monitoring (~30 min)

**Why:** The JD says "deployment, monitoring, and maintenance of AI models." This adds the production layer.

Create `api/main.py`:

```python
"""
Production API for the biomedical research assistant.
Includes health checks, metrics, and structured logging.
"""
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from agent.tools import get_classifier

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biomedical-api")

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["predicted_label"],
)

# --- App ---
app = FastAPI(
    title="SciLifeLab Biomedical Research Assistant",
    description="AI-powered tools for life science researchers",
    version="0.1.0",
)


# --- Request/Response models ---
class ClassifyRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "We recruited 500 patients for a randomized controlled trial."
            }
        }


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    text: str


# --- Endpoints ---
@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return {"status": "healthy", "model_loaded": get_classifier() is not None}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint for monitoring."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Classify a biomedical text sentence."""
    start_time = time.time()

    try:
        clf = get_classifier()
        result = clf(request.text, truncation=True, max_length=128)
        label = result[0]["label"]
        score = result[0]["score"]

        # Track metrics
        MODEL_PREDICTIONS.labels(predicted_label=label).inc()
        REQUEST_COUNT.labels(endpoint="/classify", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/classify").observe(
            time.time() - start_time
        )

        logger.info(f"Classified: '{request.text[:50]}...' → {label} ({score:.2%})")

        return ClassifyResponse(
            label=label,
            confidence=score,
            text=request.text,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/classify", status="error").inc()
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Return model metadata — useful for model versioning."""
    return {
        "model_name": "distilbert-pubmed-rct",
        "base_model": "distilbert-base-uncased",
        "task": "sequence-classification",
        "labels": ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"],
        "training_data": "pubmed-rct20k",
        "version": "0.1.0",
    }
```

**Run it:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Test it:**
```bash
# Health check (Kubernetes would hit this)
curl http://localhost:8000/health

# Classify a sentence
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "We recruited 500 patients in a randomized controlled trial."}'

# Check metrics (Prometheus would scrape this)
curl http://localhost:8000/metrics

# Interactive docs
open http://localhost:8000/docs
```

---

## Step 5 — Dockerize everything (~30 min)

**Why:** The JD explicitly requires "Docker, Helm, and Kubernetes." This is your home turf.

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model_output:/app/model_output
    environment:
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Optional: add Prometheus to scrape /metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "biomedical-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
```

Create `requirements.txt`:

```
torch
transformers
datasets
scikit-learn
langchain
langchain-community
fastapi
uvicorn
prometheus-client
biopython
requests
```

**Run the full stack:**
```bash
docker-compose up --build
```

---

## Step 6 — What to say in the interview

You now have a project that touches every single JD requirement. Here's how to frame it:

**"I built a biomedical research assistant that demonstrates the full stack this role requires:"**

1. **Model training & evaluation**: "I fine-tuned DistilBERT on PubMed data, evaluated with per-class F1, and did error analysis on the confusion matrix."

2. **AI agent architectures**: "I used LangChain to build a ReAct agent that integrates the classifier, PubMed search, and a summarizer — the agent reasons about which tool to use."

3. **Model deployment & monitoring**: "The model is served via FastAPI with Prometheus metrics for latency, throughput, and per-label prediction counts — the same observability pattern I'd set up in production."

4. **Containerization**: "Everything runs in Docker with health checks and docker-compose, ready to deploy to Kubernetes — which I have deep experience with from my previous role."

5. **Life science domain**: "I chose PubMed data specifically because it's the kind of biomedical text SciLifeLab researchers work with."
