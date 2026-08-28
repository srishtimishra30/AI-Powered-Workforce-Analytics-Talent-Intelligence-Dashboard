from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader
)

from langchain_text_splitters import MarkdownHeaderTextSplitter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DOCUMENTS_PATH = Path(
    "C:/Users/dell/OneDrive/Desktop/RAG/knowledge_base/documents"
)


# ============================================================
# CREATE STRUCTURED CHUNKS
# ============================================================

def create_chunks():

    # --------------------------------------------------------
    # LOAD MARKDOWN DOCUMENTS
    # --------------------------------------------------------

    loader = DirectoryLoader(
        str(DOCUMENTS_PATH),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()


    # --------------------------------------------------------
    # MARKDOWN HEADER SPLITTER
    # --------------------------------------------------------

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )


    # --------------------------------------------------------
    # CREATE STRUCTURED CHUNKS
    # --------------------------------------------------------

    all_chunks = []

    chunk_id = 0


    for document in documents:

        source = document.metadata["source"]

        # Extract document name
        document_name = Path(source).stem


        markdown_chunks = markdown_splitter.split_text(
            document.page_content
        )


        for section in markdown_chunks:

            chunk_id += 1

            metadata = section.metadata


            # ------------------------------------------------
            # HEADER INFORMATION
            # ------------------------------------------------

            header_1 = metadata.get(
                "Header 1",
                document_name
            )

            header_2 = metadata.get(
                "Header 2",
                ""
            )


            # ------------------------------------------------
            # ADD METADATA
            # ------------------------------------------------

            section.metadata["document_name"] = document_name

            section.metadata["source"] = source

            section.metadata["parent_section"] = header_1

            section.metadata["section"] = header_2

            section.metadata["chunk_id"] = chunk_id

            section.metadata["parent_id"] = (
                f"{document_name}_{header_1}"
            )


            # ------------------------------------------------
            # CREATE STRUCTURED CONTENT
            # ------------------------------------------------

            section_title = header_2.strip().rstrip(":") if header_2 else header_1.strip()

            structured_content = f"""Document: {header_1.strip()}
Section: {section_title}

{section.page_content.strip()}""".strip()


            # ------------------------------------------------
            # UPDATE CONTENT
            # ------------------------------------------------

            section.page_content = structured_content


            # ------------------------------------------------
            # STORE CHUNK
            # ------------------------------------------------

            all_chunks.append(section)


    return all_chunks


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    chunks = create_chunks()


    print("\n")
    print("=" * 70)
    print("STRUCTURED CHUNKS")
    print("=" * 70)


    for chunk in chunks:

        print("\n" + "=" * 70)

        print(
            f"CHUNK {chunk.metadata['chunk_id']}"
        )

        print("\nMetadata:")

        print(chunk.metadata)

        print("\nContent:")

        print(chunk.page_content)

        print(
            f"\nCharacters: "
            f"{len(chunk.page_content)}"
        )


    print("\n")

    print("=" * 70)

    print(
        f"TOTAL CHUNKS: {len(chunks)}"
    )

    print("=" * 70)