from sentence_transformers import CrossEncoder


# ============================================================
# RERANKER MODEL
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("=" * 70)
print("LOADING RERANKER")
print("=" * 70)

reranker = CrossEncoder(MODEL_NAME)

print("Reranker loaded successfully.")


# ============================================================
# RERANK FUNCTION
# ============================================================

def rerank_documents(query, documents, top_k=3):

    if not documents:
        return []

    # --------------------------------------------------------
    # Create query-document pairs
    # --------------------------------------------------------

    pairs = []

    for document in documents:

        pairs.append(
            (
                query,
                document.page_content
            )
        )

    # --------------------------------------------------------
    # Calculate relevance scores
    # --------------------------------------------------------

    scores = reranker.predict(pairs)

    # --------------------------------------------------------
    # Attach scores to documents
    # --------------------------------------------------------

    ranked_documents = []

    for document, score in zip(documents, scores):

        ranked_documents.append(
            (
                float(score),
                document
            )
        )

    # --------------------------------------------------------
    # Sort by reranker score
    # Higher = more relevant
    # --------------------------------------------------------

    ranked_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Return top K
    # --------------------------------------------------------

    return ranked_documents[:top_k]