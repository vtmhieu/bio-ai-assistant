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
                f"PMID: {pmid}\n" f"Title: {title}\n" f"Abstract: {abstract}...\n"
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
