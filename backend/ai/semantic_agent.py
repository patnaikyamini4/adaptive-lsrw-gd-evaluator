from typing import Any


MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_semantic_model():
    """
    Lazy load the SentenceTransformer model on first call.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def calculate_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Calculate raw cosine semantic similarity between two texts.
    """
    from sentence_transformers import util

    model = get_semantic_model()
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    similarity = util.cos_sim(emb_a, emb_b).item()
    return round(float(similarity), 4)


def evaluate_semantic_similarity(
    golden_answer: str,
    candidate_answer: str,
) -> dict[str, Any]:
    """
    Evaluate semantic similarity of candidate answer against golden answer.
    Returns model name and raw similarity without score thresholds.
    """
    similarity = calculate_semantic_similarity(golden_answer, candidate_answer)
    return {
        "model": MODEL_NAME,
        "similarity": similarity,
    }