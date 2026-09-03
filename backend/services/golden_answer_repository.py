from datetime import datetime, timezone

from backend.services.mongodb import db


collection = db["golden_answers"]


def save_golden_answer(
    question,
    expert_answer_1,
    expert_answer_2,
    golden_answer,
    key_points
):
    """
    Save a generated Golden Answer to MongoDB.
    """

    document = {
        "question": question,
        "expert_answers": {
            "expert_answer_1": expert_answer_1,
            "expert_answer_2": expert_answer_2
        },
        "golden_answer": golden_answer,
        "key_points": key_points,
        "model": "qwen/qwen3.8-max",
        "created_at": datetime.now(timezone.utc)
    }

    result = collection.insert_one(document)

    return str(result.inserted_id)