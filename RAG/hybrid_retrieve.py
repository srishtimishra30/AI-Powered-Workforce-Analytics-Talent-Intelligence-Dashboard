"""
Lightweight hybrid retrieval using TF-IDF + FAISS.

Replaces:
  - sentence-transformers / HuggingFaceEmbeddings  (1.5 GB PyTorch models)
  - langchain-chroma / chromadb                    (~100 MB install)

With:
  - sklearn TfidfVectorizer  (pure Python, already in scikit-learn)
  - faiss-cpu                (~7 MB)
  - rank_bm25                (pure Python, unchanged)

This keeps deployment size on Render well within the free-tier limits.
"""

from pathlib import Path

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

from chunk_documents import create_chunks
from reranker import rerank_documents
from context_expander import expand_context

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Build knowledge base at module load time (happens once at app startup)
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING KNOWLEDGE BASE")
print("=" * 70)

chunks = create_chunks()

document_names = {chunk.metadata.get("document_name") for chunk in chunks}

print(f"Documents loaded: {len(document_names)}")
print(f"Total chunks loaded: {len(chunks)}")

# ---------------------------------------------------------------------------
# TF-IDF vectoriser (used for semantic-ish search via cosine similarity)
# ---------------------------------------------------------------------------
print("\n")
print("=" * 70)
print("BUILDING TF-IDF INDEX")
print("=" * 70)

corpus = [chunk.page_content for chunk in chunks]

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams for better recall
    sublinear_tf=True,
    min_df=1,
)

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)   # sparse (n_chunks, vocab)

# Normalise rows so dot-product == cosine similarity
tfidf_dense = tfidf_matrix.toarray().astype("float32")
norms = np.linalg.norm(tfidf_dense, axis=1, keepdims=True)
norms[norms == 0] = 1.0
tfidf_dense_norm = tfidf_dense / norms

# Build a flat FAISS index (exact cosine search via inner-product on unit vectors)
dim = tfidf_dense_norm.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(tfidf_dense_norm)

print(f"FAISS index built. Vectors: {faiss_index.ntotal}, Dim: {dim}")

# ---------------------------------------------------------------------------
# BM25 index (keyword retrieval, unchanged algorithm)
# ---------------------------------------------------------------------------
tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created.")


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def semantic_retrieve(query: str, k: int = 5):
    """
    TF-IDF + FAISS cosine-similarity retrieval.
    Returns list of (document, score) tuples — same signature as the old
    Chroma-based version so the rest of the pipeline is unaffected.
    """
    query_vec = tfidf_vectorizer.transform([query]).toarray().astype("float32")
    q_norm = np.linalg.norm(query_vec)
    if q_norm > 0:
        query_vec = query_vec / q_norm

    scores, indices = faiss_index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((chunks[idx], float(score)))

    return results


def keyword_retrieve(query: str, k: int = 5):
    """BM25 keyword retrieval. Returns list of (score, document) tuples."""
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    for index in ranked_indexes[:k]:
        results.append((float(scores[index]), chunks[index]))

    return results


def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k: int = 60
):
    """
    Combine ranked lists from semantic and keyword search using RRF.
    semantic_results: list of (document, distance)
    keyword_results:  list of (score,    document)
    Returns: list of (fused_score, document) sorted descending.
    """
    fused_scores = {}
    documents = {}

    for rank, (document, distance) in enumerate(semantic_results, start=1):
        doc_id = document.metadata["chunk_id"]
        documents[doc_id] = document
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, (score, document) in enumerate(keyword_results, start=1):
        doc_id = document.metadata["chunk_id"]
        documents[doc_id] = document
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return [(float(score), documents[doc_id]) for doc_id, score in ranked]


def hybrid_retrieve(
    query: str,
    retrieval_k: int = 5,
    rrf_k: int = 10,
    final_k: int = 3
):
    """End-to-end hybrid retrieval: semantic + BM25 → RRF → rerank → expand."""

    semantic_results = semantic_retrieve(query, k=retrieval_k)
    keyword_results  = keyword_retrieve(query, k=retrieval_k)

    fused_results = reciprocal_rank_fusion(semantic_results, keyword_results)

    candidate_results   = fused_results[:rrf_k]
    candidate_documents = [doc for _score, doc in candidate_results]

    reranked_results   = rerank_documents(
        query=query,
        documents=candidate_documents,
        top_k=final_k
    )
    reranked_documents = [doc for _score, doc in reranked_results]

    expanded_context = expand_context(
        ranked_documents=reranked_documents,
        all_chunks=chunks,
        window_size=1
    )

    final_context_str = "\n\n---\n\n".join(
        [doc.page_content.strip() for doc in expanded_context]
    )

    return {
        "query":         query,
        "semantic":      semantic_results,
        "keyword":       keyword_results,
        "rrf":           candidate_results,
        "reranked":      reranked_results,
        "expanded":      expanded_context,
        "final_context": final_context_str,
    }
