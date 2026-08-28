from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from chunk_documents import create_chunks
from reranker import rerank_documents
from context_expander import expand_context


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("=" * 70)
print("LOADING KNOWLEDGE BASE")
print("=" * 70)

chunks = create_chunks()

document_names = {
    chunk.metadata.get("document_name")
    for chunk in chunks
}

print(f"Documents loaded: {len(document_names)}")
print(f"Total chunks loaded: {len(chunks)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# LOAD CHROMA VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_PATH),
    embedding_function=embeddings,
    collection_name="workforce_knowledge"
)

print("Chroma vector database loaded.")


# ============================================================
# CREATE BM25 INDEX
# ============================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created.")


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(query, k=5):

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    # Chroma returns:
    # (Document, distance)

    return results


# ============================================================
# BM25 / KEYWORD RETRIEVAL
# ============================================================

def keyword_retrieve(query, k=5):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indexes[:k]:

        # Standardized structure:
        # (score, Document)

        results.append(
            (
                float(scores[index]),
                chunks[index]
            )
        )

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    fused_scores = {}
    documents = {}

    # ========================================================
    # SEMANTIC RANKING
    # ========================================================

    # Semantic structure:
    # (Document, distance)

    for rank, (document, distance) in enumerate(
        semantic_results,
        start=1
    ):

        doc_id = document.metadata["chunk_id"]

        documents[doc_id] = document

        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

    # ========================================================
    # BM25 RANKING
    # ========================================================

    # BM25 structure:
    # (score, Document)

    for rank, (score, document) in enumerate(
        keyword_results,
        start=1
    ):

        doc_id = document.metadata["chunk_id"]

        documents[doc_id] = document

        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

    # ========================================================
    # SORT FUSED RESULTS
    # ========================================================

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc_id, score in ranked:

        results.append(
            (
                float(score),
                documents[doc_id]
            )
        )

    return results


# ============================================================
# HYBRID + RRF + RERANKER + CONTEXT EXPANSION
# ============================================================

def hybrid_retrieve(
    query,
    retrieval_k=5,
    rrf_k=10,
    final_k=3
):

    # ========================================================
    # STEP 1: SEMANTIC RETRIEVAL
    # ========================================================

    semantic_results = semantic_retrieve(
        query,
        k=retrieval_k
    )


    # ========================================================
    # STEP 2: BM25 RETRIEVAL
    # ========================================================

    keyword_results = keyword_retrieve(
        query,
        k=retrieval_k
    )


    # ========================================================
    # STEP 3: RRF FUSION
    # ========================================================

    fused_results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results
    )


    # ========================================================
    # STEP 4: SELECT TOP RRF CANDIDATES
    # ========================================================

    candidate_results = fused_results[:rrf_k]


    # ========================================================
    # STEP 5: EXTRACT DOCUMENTS FOR RERANKER
    # ========================================================

    candidate_documents = [
        document
        for score, document in candidate_results
    ]


    # ========================================================
    # STEP 6: CROSS-ENCODER RERANKING
    # ========================================================

    reranked_results = rerank_documents(
        query=query,
        documents=candidate_documents,
        top_k=final_k
    )


    # ========================================================
    # STEP 7: EXTRACT RERANKED DOCUMENTS
    # ========================================================

    reranked_documents = [
        document
        for score, document in reranked_results
    ]


    # ========================================================
    # STEP 8: CONTEXT EXPANSION
    # ========================================================

    expanded_context = expand_context(
        ranked_documents=reranked_documents,
        all_chunks=chunks,
        window_size=1
    )


    # ========================================================
    # STEP 9: FORMAT FINAL CONTEXT FOR LLM
    # ========================================================

    final_context_str = "\n\n---\n\n".join(
        [doc.page_content.strip() for doc in expanded_context]
    )


    # ========================================================
    # RETURN ALL RESULTS
    # ========================================================

    return {
        "query": query,
        "semantic": semantic_results,
        "keyword": keyword_results,
        "rrf": candidate_results,
        "reranked": reranked_results,
        "expanded": expanded_context,
        "final_context": final_context_str
    }


# ============================================================
# DISPLAY SEMANTIC RESULTS
# ============================================================

def display_semantic_results(
    results,
    title="SEMANTIC RETRIEVAL"
):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, (document, distance) in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 70)

        print(f"Rank {rank}")

        print(f"Distance: {distance:.4f}")

        print("\nChunk ID:")
        print(document.metadata.get("chunk_id"))

        print("\nDocument:")
        print(
            document.metadata.get("document_name")
        )

        print("\nSection:")
        print(
            document.metadata.get("section")
        )

        print("\nContent:")
        print(document.page_content)


# ============================================================
# DISPLAY SCORED RESULTS
# ============================================================

def display_results(
    title,
    results,
    score_name
):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, (score, document) in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 70)

        print(f"Rank {rank}")

        print(f"{score_name}: {score:.4f}")

        print("\nChunk ID:")
        print(document.metadata.get("chunk_id"))

        print("\nDocument:")
        print(
            document.metadata.get("document_name")
        )

        print("\nSection:")
        print(
            document.metadata.get("section")
        )

        print("\nContent:")
        print(document.page_content)


# ============================================================
# DISPLAY EXPANDED CONTEXT
# ============================================================

def display_expanded_context(documents):

    print("\n")
    print("=" * 70)
    print("EXPANDED CONTEXT")
    print("=" * 70)

    for rank, document in enumerate(
        documents,
        start=1
    ):

        print("\n" + "-" * 70)

        print(f"CONTEXT CHUNK {rank}")

        print("\nChunk ID:")
        print(
            document.metadata.get("chunk_id")
        )

        print("\nDocument:")
        print(
            document.metadata.get("document_name")
        )

        print("\nSection:")
        print(
            document.metadata.get("section")
        )

        print("\nContent:")
        print(document.page_content)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = "What are the guidelines for employee workload?"


    # ========================================================
    # DISPLAY QUERY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("QUERY")
    print("=" * 70)

    print(f"\n{query}")


    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================

    results = hybrid_retrieve(
        query=query,
        retrieval_k=5,
        rrf_k=10,
        final_k=3
    )


    # ========================================================
    # DISPLAY SEMANTIC RESULTS
    # ========================================================

    display_semantic_results(
        results["semantic"]
    )


    # ========================================================
    # DISPLAY BM25 RESULTS
    # ========================================================

    display_results(
        "KEYWORD / BM25 RETRIEVAL",
        results["keyword"],
        "BM25 Score"
    )


    # ========================================================
    # DISPLAY RRF RESULTS
    # ========================================================

    display_results(
        "RRF FUSED RETRIEVAL",
        results["rrf"],
        "RRF Score"
    )


    # ========================================================
    # DISPLAY RERANKED RESULTS
    # ========================================================

    display_results(
        "RERANKED RETRIEVAL",
        results["reranked"],
        "Reranker Score"
    )


    # ========================================================
    # DISPLAY EXPANDED CONTEXT
    # ========================================================

    display_expanded_context(
        results["expanded"]
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("HYBRID + RRF + RERANKER + CONTEXT EXPANSION COMPLETE")
    print("=" * 70)