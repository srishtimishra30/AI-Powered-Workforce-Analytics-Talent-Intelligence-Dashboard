from typing import List

from langchain_core.documents import Document

def expand_context(
    ranked_documents: List[Document],
    all_chunks: List[Document],
    window_size: int = 1
) -> List[Document]:
    document_chunks = {}

    for chunk in all_chunks:

        document_name = chunk.metadata.get(
            "document_name"
        )

        if document_name not in document_chunks:

            document_chunks[document_name] = []

        document_chunks[document_name].append(chunk)

    for document_name in document_chunks:

        document_chunks[document_name].sort(
            key=lambda chunk: int(
                chunk.metadata.get("chunk_id", 0)
            )
        )

    expanded_chunks = {}

    for ranked_document in ranked_documents:

        document_name = ranked_document.metadata.get(
            "document_name"
        )

        chunk_id = ranked_document.metadata.get(
            "chunk_id"
        )

        if document_name not in document_chunks:
            continue

        chunks = document_chunks[document_name]


        target_index = None

        for index, chunk in enumerate(chunks):

            if chunk.metadata.get("chunk_id") == chunk_id:

                target_index = index
                break

        if target_index is None:
            continue

        start_index = max(
            0,
            target_index - window_size
        )

        end_index = min(
            len(chunks),
            target_index + window_size + 1
        )


        for chunk in chunks[
            start_index:end_index
        ]:

            current_chunk_id = chunk.metadata.get(
                "chunk_id"
            )

            if current_chunk_id not in expanded_chunks:

                expanded_chunks[
                    current_chunk_id
                ] = chunk

    expanded_context = sorted(
        expanded_chunks.values(),
        key=lambda chunk: int(
            chunk.metadata.get("chunk_id", 0)
        )
    )

    return expanded_context
