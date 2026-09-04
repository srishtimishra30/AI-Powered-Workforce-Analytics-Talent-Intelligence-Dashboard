from hybrid_retrieve import hybrid_retrieve
from llm import generate_answer

def run_rag_pipeline(
    question: str,
    model_name: str = "openai/gpt-oss-20b",
    verbose: bool = True
) -> str:
    """
    Execute end-to-end RAG pipeline for a given user question.

    Flow:
    User Question
         ↓
    Hybrid Retrieval (Semantic + BM25)
         ↓
    Reciprocal Rank Fusion (RRF)
         ↓
    Cross-Encoder Reranking
         ↓
    Context Expansion
         ↓
    Ollama LLM Answer Generation
         ↓
    Final Answer
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)
    retrieval_output = hybrid_retrieve(
        query=question,
        retrieval_k=5,
        rrf_k=10,
        final_k=3
    )

    final_context = retrieval_output["final_context"]

    if verbose:
        print("\n" + "-" * 70)
        print("FINAL RETRIEVED CONTEXT (passed to LLM):")
        print("-" * 70)
        print(final_context)
    answer = generate_answer(
        question=question,
        context=final_context,
        model_name=model_name
    )

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(answer)

    return answer

if __name__ == "__main__":

    test_questions = [
        "What are the guidelines for employee workload?",
        "What actions should be taken for career growth?",
        "What are the performance evaluation indicators?",
        "What is the policy for remote work?"
    ]

    print("=" * 70)
    print("WORKFORCE ANALYTICS RAG PIPELINE — END-TO-END TEST")
    print("=" * 70)

    for idx, q in enumerate(test_questions, start=1):
        print(f"\n\n{'#' * 70}")
        print(f"TEST CASE {idx}")
        print(f"{'#' * 70}")
        run_rag_pipeline(q)
