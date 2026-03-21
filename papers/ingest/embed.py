import torch
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL, EMBED_BATCH

_model: SentenceTransformer | None = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model() -> SentenceTransformer:
    """Load model once and cache. Called explicitly so startup message appears early."""
    global _model
    if _model is None:
        device = _get_device()
        print(f"Loading model: {EMBED_MODEL}  (downloads ~270 MB on first run)")
        print(f"Device: {device}")
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
        print("Model ready.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = load_model()
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,  # unit vectors → cosine sim = dot product at query time
        show_progress_bar=False,
    )
    return vectors.tolist()
