from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("=" * 70)
print("LOADING RERANKER")
print("=" * 70)

reranker = CrossEncoder(MODEL_NAME)

print("Reranker loaded successfully.")

def rerank_documents(query, documents, top_k=3):

    if not documents:
        return []
    pairs = []

    for document in documents:

        pairs.append(
            (
                query,
                document.page_content
            )
        )

    scores = reranker.predict(pairs)

    ranked_documents = []

    for document, score in zip(documents, scores):

        ranked_documents.append(
            (
                float(score),
                document
            )
        )

    ranked_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_documents[:top_k]
