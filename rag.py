from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

vector_index = None
document_chunks = []


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_vector_store(text):

    global vector_index
    global document_chunks

    document_chunks = chunk_text(text)

    embeddings = embedding_model.encode(
        document_chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    vector_index = faiss.IndexFlatL2(dimension)

    vector_index.add(
        embeddings.astype("float32")
    )


def retrieve_context(query, top_k=4):

    if vector_index is None:
        return ""

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = vector_index.search(
        query_embedding.astype("float32"),
        top_k
    )

    retrieved = []

    for idx in indices[0]:
        if idx < len(document_chunks):
            retrieved.append(document_chunks[idx])

    return "\n\n".join(retrieved)