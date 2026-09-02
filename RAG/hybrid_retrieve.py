from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from chunk_documents import create_chunks
from reranker import rerank_documents
from context_expander import expand_context

BASE_DIR = Path(__file__).parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

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

print("\n")
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_PATH),
    embedding_function=embeddings,
    collection_name="workforce_knowledge"
)

print("Chroma vector database loaded.")

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created.")

def semantic_retrieve(query, k=5):

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    return results


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

        results.append(
            (
                float(scores[index]),
                chunks[index]
            )
        )

    return results

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    fused_scores = {}
    documents = {}

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

def hybrid_retrieve(
    query,
    retrieval_k=5,
    rrf_k=10,
    final_k=3
):

    semantic_results = semantic_retrieve(
        query,
        k=retrieval_k
    )

    keyword_results = keyword_retrieve(
        query,
        k=retrieval_k
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results
    )

    candidate_results = fused_results[:rrf_k]

    candidate_documents = [
        document
        for score, document in candidate_results
    ]


    reranked_results = rerank_documents(
        query=query,
        documents=candidate_documents,
        top_k=final_k
    )


    reranked_documents = [
        document
        for score, document in reranked_results
    ]

    expanded_context = expand_context(
        ranked_documents=reranked_documents,
        all_chunks=chunks,
        window_size=1
    )
    final_context_str = "\n\n---\n\n".join(
        [doc.page_content.strip() for doc in expanded_context]
    )

    return {
        "query": query,
        "semantic": semantic_results,
        "keyword": keyword_results,
        "rrf": candidate_results,
        "reranked": reranked_results,
        "expanded": expanded_context,
        "final_context": final_context_str
    }

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


if __name__ == "__main__":

    query = "What are the guidelines for employee workload?"

    print("\n")
    print("=" * 70)
    print("QUERY")
    print("=" * 70)

    print(f"\n{query}")


    results = hybrid_retrieve(
        query=query,
        retrieval_k=5,
        rrf_k=10,
        final_k=3
    )

    display_semantic_results(
        results["semantic"]
    )

    display_results(
        "KEYWORD / BM25 RETRIEVAL",
        results["keyword"],
        "BM25 Score"
    )

    display_results(
        "RRF FUSED RETRIEVAL",
        results["rrf"],
        "RRF Score"
    )

    display_results(
        "RERANKED RETRIEVAL",
        results["reranked"],
        "Reranker Score"
    )

    display_expanded_context(
        results["expanded"]
    )



    print("\n")
    print("=" * 70)
    print("HYBRID + RRF + RERANKER + CONTEXT EXPANSION COMPLETE")
    print("=" * 70)