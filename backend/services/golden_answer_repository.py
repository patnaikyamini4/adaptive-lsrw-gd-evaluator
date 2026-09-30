from datetime import datetime, timezone

from backend.services.mongodb import db


collection = db["golden_answers"]
golden_answers_collection = collection


def save_golden_answer(
    question,
    expert_answer_1,
    expert_answer_2,
    golden_answer,
    key_points,
    question_id=None,
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

    if question_id is not None:
        document["question_id"] = question_id

    result = collection.insert_one(document)

    return str(result.inserted_id)


def get_golden_answer(question_id):
    """
    Retrieve a Golden Answer document by question_id from MongoDB.

    Returns the document dictionary without MongoDB's '_id', or None if not found.
    """
    document = collection.find_one({"question_id": question_id})

    if document is None:
        return None

    document_copy = document.copy()
    document_copy.pop("_id", None)
    return document_copy
