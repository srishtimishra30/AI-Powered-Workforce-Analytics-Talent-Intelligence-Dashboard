"""
Lightweight cosine-similarity reranker using sklearn's TfidfVectorizer.

Replaces:
  - sentence-transformers CrossEncoder  (downloads PyTorch + cross-encoder
    model ~100 MB, requires ~1.5 GB PyTorch install)

With:
  - sklearn cosine_similarity            (pure Python, zero extra install,
    already used by hybrid_retrieve.py)

The reranker scores each (query, chunk) pair by TF-IDF cosine similarity
and re-sorts the candidates, giving a fast and dependency-free re-ranking
step that is good enough for a small workforce knowledge base.
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 3
) -> List[tuple]:
    """
    Re-rank *documents* against *query* using TF-IDF cosine similarity.

    Parameters
    ----------
    query     : the user's question
    documents : candidate documents from RRF fusion
    top_k     : how many top documents to return

    Returns
    -------
    List of (score: float, document: Document) sorted highest-score-first,
    capped at top_k.
    """
    if not documents:
        return []

    corpus = [query] + [doc.page_content for doc in documents]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Row 0 is the query; rows 1..N are the documents
    query_vec = tfidf_matrix[0]
    doc_vecs   = tfidf_matrix[1:]

    scores = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked = sorted(
        zip(scores.tolist(), documents),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]
