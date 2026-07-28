import numpy as np

from rag_studio.schema import Chunk
from rag_studio.ingestion.vector_store import FaissVectorStore


def test_vector_store_applies_metadata_filter_after_similarity_search() -> None:
    store = FaissVectorStore()
    chunks = [
        Chunk(id="resume", text="resume chunk", metadata={"doc_type": "resume"}),
        Chunk(id="job", text="job chunk", metadata={"doc_type": "job_description"}),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype="float32",
    )
    store.add(chunks, vectors)

    results = store.search(
        np.array([1.0, 0.0], dtype="float32"),
        top_k=1,
        metadata_filter={"doc_type": "job_description"},
    )

    assert [result.chunk.id for result in results] == ["job"]
