from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

DOCUMENTS_PATH = Path(__file__).parent / "knowledge_base" / "documents"


loader = DirectoryLoader(
    path = str(DOCUMENTS_PATH),
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)

documents = loader.load()


print(f"Number of documents loaded: {len(documents)}")

for i, document in enumerate(documents, start=1):
    print("\n" + "=" * 60)
    print(f"DOCUMENT {i}")
    print("=" * 60)

    print("Source:", document.metadata["source"])
    print("Content:")
    print(document.page_content[:500])

