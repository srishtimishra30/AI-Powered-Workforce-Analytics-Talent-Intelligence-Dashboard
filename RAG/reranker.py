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
    if not documents:
        return []

    corpus = [query] + [doc.page_content for doc in documents]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = tfidf_matrix[0]
    doc_vecs   = tfidf_matrix[1:]

    scores = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked = sorted(
        zip(scores.tolist(), documents),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]
