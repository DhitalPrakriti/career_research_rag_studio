import pytest

from rag_studio.ragas_eval_cli import SentenceTransformerLangchainEmbeddings


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts, **kwargs):  # pragma: no cover - not exercised here
        return [[0.0] * 3 for _ in texts]


@pytest.fixture
def embeddings(monkeypatch: pytest.MonkeyPatch) -> SentenceTransformerLangchainEmbeddings:
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    return SentenceTransformerLangchainEmbeddings("some/embedding-model")


def test_model_attribute_is_a_string(
    embeddings: SentenceTransformerLangchainEmbeddings,
) -> None:
    """RAGAS validates getattr(embeddings, "model") as Optional[str].

    Exposing the encoder object here raises a pydantic ValidationError inside RAGAS
    telemetry, which silently nulls out every embedding-based metric rather than
    failing loudly. Guard the contract.
    """
    model = getattr(embeddings, "model", None)

    assert isinstance(model, str)
    assert model == "some/embedding-model"


def test_encoder_is_kept_separately(
    embeddings: SentenceTransformerLangchainEmbeddings,
) -> None:
    assert isinstance(embeddings._encoder, _FakeSentenceTransformer)
    assert embeddings.model_name == "some/embedding-model"
