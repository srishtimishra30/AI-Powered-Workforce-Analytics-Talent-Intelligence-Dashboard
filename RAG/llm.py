import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load .env from the repo root (two levels up from RAG/llm.py)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


PROMPT_TEMPLATE = """You are an AI assistant for workforce analytics.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say that the
information is not available in the knowledge base.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def query_groq(
    question: str,
    context: str,
    model_name: str = "openai/gpt-oss-20b"
) -> Optional[str]:
    """
    Query the Groq API (free tier).
    Uses openai/gpt-oss-20b — available on the free Groq plan.
    Sign up at https://console.groq.com to get a free API key.
    """
    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        client = Groq(api_key=api_key)
        formatted_prompt = prompt.format(
            context=context,
            question=question
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM Notice] Groq API error: {e}")
        return None


def generate_answer(
    question: str,
    context: str,
    model_name: str = "openai/gpt-oss-20b",
    **kwargs
) -> str:
    """
    Generate a grounded answer for the question using the retrieved context.
    Uses Groq API (free tier — llama-3.3-70b-versatile).
    """
    answer = query_groq(question, context, model_name)
    if answer:
        return answer

    return (
        "I could not generate an answer because the GROQ_API_KEY is not configured "
        "or the Groq API is unavailable. "
        "Please set GROQ_API_KEY (free at https://console.groq.com) "
        "in your environment variables."
    )


if __name__ == "__main__":
    sample_context = (
        "Document: Employee Retention Guidelines\n"
        "Section: Retention Risk Indicators\n\n"
        "1. High workload\nEmployees consistently working excessive hours "
        "may experience higher levels of stress and reduced job satisfaction."
    )
    sample_question = "What causes reduced job satisfaction?"

    print("Testing LLM generation...")
    res = generate_answer(sample_question, sample_context)
    print("\nAnswer:")
    print(res)
