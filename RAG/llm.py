import json
import os
import urllib.request
from typing import Optional

from langchain_core.prompts import PromptTemplate


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


def query_ollama(
    formatted_prompt: str,
    model_name: str = "llama3",
    host: str = "http://localhost:11434"
) -> Optional[str]:
    """
    Query local Ollama server via REST API endpoint.
    Returns None if Ollama isn't running (e.g. on a deployed server) -
    generate_answer() will then fall back to the Claude API below.
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": formatted_prompt,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception:
        # Ollama server unreachable (normal on a deployed server - no
        # local machine to run Ollama on) or model not loaded
        return None


def query_claude_fallback(question: str, context: str) -> str:
    """
    Fallback LLM generator using the Anthropic Claude API.
    Used automatically whenever Ollama isn't reachable - this is what
    runs on a deployed server, since it doesn't need a large local
    model or a GPU, just a lightweight web request.
    """
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("Information is not available - ANTHROPIC_API_KEY is not "
                "set, so the fallback answer generator cannot run.")

    client = Anthropic(api_key=api_key)

    formatted_prompt = prompt.format(context=context, question=question)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    return response.content[0].text.strip()


def generate_answer(
    question: str,
    context: str,
    model_name: str = "llama3",
    ollama_host: str = "http://localhost:11434"
) -> str:
    """
    Generate a grounded answer for the question using context.
    Tries Ollama first (free, local, great for development). If that's
    not reachable - which will always be the case on a deployed server
    with no Ollama installed - falls back to the Claude API instead.
    """
    formatted_prompt = prompt.format(context=context, question=question)

    answer = query_ollama(
        formatted_prompt=formatted_prompt,
        model_name=model_name,
        host=ollama_host
    )

    if answer:
        return answer
    print(f"\n[LLM Notice] Could not connect to Ollama at {ollama_host}.")
    print("[LLM Notice] Using Claude API fallback to generate answer...\n")
    return query_claude_fallback(question, context)


if __name__ == "__main__":
    sample_context = "Document: Employee Retention Guidelines\nSection: Retention Risk Indicators\n\n1. High workload\nEmployees consistently working excessive hours may experience higher levels of stress and reduced job satisfaction."
    sample_question = "What causes reduced job satisfaction?"

    print("Testing LLM generation...")
    res = generate_answer(sample_question, sample_context)
    print("\nAnswer:")
    print(res)