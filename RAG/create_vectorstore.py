import shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from chunk_documents import create_chunks



# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


# ============================================================
# LOAD STRUCTURED CHUNKS
# ============================================================

print("=" * 70)
print("LOADING STRUCTURED CHUNKS")
print("=" * 70)

chunks = create_chunks()

print(f"Total chunks loaded: {len(chunks)}")


# ============================================================
# DISPLAY CHUNK INFORMATION
# ============================================================

for chunk in chunks:

    print(f"\nChunk ID: {chunk.metadata.get('chunk_id')}")
    print(f"Parent ID: {chunk.metadata.get('parent_id')}")
    print(f"Document: {chunk.metadata.get('document_name')}")
    print(f"Section: {chunk.metadata.get('section')}")


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
# CREATE CHROMA VECTOR DATABASE
# ============================================================

print("\n")
print("=" * 70)
print("CREATING CHROMA VECTOR DATABASE")
print("=" * 70)

if VECTORSTORE_PATH.exists():
    shutil.rmtree(VECTORSTORE_PATH)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTORSTORE_PATH),
    collection_name="workforce_knowledge"
)


# ============================================================
# VERIFY
# ============================================================

print("\n")
print("=" * 70)
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Location: {VECTORSTORE_PATH}")
print(f"Documents embedded: {len(chunks)}")


# ============================================================
# TEST RETRIEVAL
# ============================================================

test_query = "What are the guidelines for employee workload?"

print("\n")
print("=" * 70)
print("TEST RETRIEVAL")
print("=" * 70)

print(f"\nQuery: {test_query}")


results = vectorstore.similarity_search(
    test_query,
    k=3
)


for rank, document in enumerate(results, start=1):

    print("\n" + "-" * 70)
    print(f"RESULT {rank}")

    print("\nChunk ID:")
    print(document.metadata.get("chunk_id"))

    print("\nParent ID:")
    print(document.metadata.get("parent_id"))

    print("\nDocument:")
    print(document.metadata.get("document_name"))

    print("\nSection:")
    print(document.metadata.get("section"))

    print("\nContent:")
    print(document.page_content)


print("\n")
print("=" * 70)
print("VECTOR STORE TEST COMPLETE")
print("=" * 70)

