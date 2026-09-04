"""
Build and persist the TF-IDF + FAISS index.

Replaces the old Chroma / HuggingFace-Embeddings vectorstore builder.
Run this script once locally (or as a build step) to regenerate the index
whenever the knowledge-base documents change:

    python RAG/create_vectorstore.py

Outputs written to RAG/vectorstore/:
    tfidf_vectorizer.pkl   — fitted TfidfVectorizer
    faiss_index.bin        — FAISS IndexFlatIP (cosine via normalised vectors)
    chunks_metadata.pkl    — serialised list of LangChain Document objects
"""

import shutil
import pickle
from pathlib import Path

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from chunk_documents import create_chunks

BASE_DIR          = Path(__file__).parent
VECTORSTORE_PATH  = BASE_DIR / "vectorstore"

# ---------------------------------------------------------------------------
# 1. Load documents and create chunks
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING STRUCTURED CHUNKS")
print("=" * 70)

chunks = create_chunks()
print(f"Total chunks loaded: {len(chunks)}")

for chunk in chunks:
    print(
        f"  Chunk {chunk.metadata.get('chunk_id'):>3} | "
        f"{chunk.metadata.get('document_name')} / "
        f"{chunk.metadata.get('section') or chunk.metadata.get('parent_section')}"
    )

# ---------------------------------------------------------------------------
# 2. Fit TF-IDF vectoriser
# ---------------------------------------------------------------------------
print("\n")
print("=" * 70)
print("FITTING TF-IDF VECTORISER")
print("=" * 70)

corpus = [chunk.page_content for chunk in chunks]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)

tfidf_matrix = vectorizer.fit_transform(corpus)
print(f"Vocabulary size  : {len(vectorizer.vocabulary_)}")
print(f"Matrix shape     : {tfidf_matrix.shape}")

# ---------------------------------------------------------------------------
# 3. Build FAISS index (cosine similarity via unit-norm inner product)
# ---------------------------------------------------------------------------
print("\n")
print("=" * 70)
print("BUILDING FAISS INDEX")
print("=" * 70)

dense = tfidf_matrix.toarray().astype("float32")
norms = np.linalg.norm(dense, axis=1, keepdims=True)
norms[norms == 0] = 1.0
dense_norm = dense / norms

dim         = dense_norm.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(dense_norm)

print(f"FAISS index built — {faiss_index.ntotal} vectors, dim {dim}")

# ---------------------------------------------------------------------------
# 4. Persist artefacts
# ---------------------------------------------------------------------------
print("\n")
print("=" * 70)
print("SAVING ARTEFACTS")
print("=" * 70)

if VECTORSTORE_PATH.exists():
    shutil.rmtree(VECTORSTORE_PATH)
VECTORSTORE_PATH.mkdir(parents=True)

vectorizer_path = VECTORSTORE_PATH / "tfidf_vectorizer.pkl"
faiss_path      = VECTORSTORE_PATH / "faiss_index.bin"
chunks_path     = VECTORSTORE_PATH / "chunks_metadata.pkl"

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

faiss.write_index(faiss_index, str(faiss_path))

with open(chunks_path, "wb") as f:
    pickle.dump(chunks, f)

print(f"TF-IDF vectoriser  → {vectorizer_path}")
print(f"FAISS index        → {faiss_path}")
print(f"Chunk metadata     → {chunks_path}")

# ---------------------------------------------------------------------------
# 5. Quick smoke test
# ---------------------------------------------------------------------------
print("\n")
print("=" * 70)
print("SMOKE TEST — RETRIEVAL")
print("=" * 70)

test_query = "What are the guidelines for employee workload?"
print(f"Query: {test_query}\n")

q_vec   = vectorizer.transform([test_query]).toarray().astype("float32")
q_norm  = np.linalg.norm(q_vec)
if q_norm > 0:
    q_vec = q_vec / q_norm

scores, indices = faiss_index.search(q_vec, 3)

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    doc = chunks[idx]
    print(f"Rank {rank}  (score={score:.4f})")
    print(f"  Document : {doc.metadata.get('document_name')}")
    print(f"  Section  : {doc.metadata.get('section') or doc.metadata.get('parent_section')}")
    print(f"  Preview  : {doc.page_content[:120].replace(chr(10), ' ')}...")
    print()

print("=" * 70)
print("VECTOR STORE BUILD COMPLETE")
print("=" * 70)
