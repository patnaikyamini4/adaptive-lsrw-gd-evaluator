from backend.ai.golden_answer_agent import generate_golden_answer
from backend.services.golden_answer_repository import (
    get_golden_answer as repo_get_golden_answer,
    save_golden_answer,
)


def get_golden_answer(question_id):
    """
    Retrieve a Golden Answer document for a specific question.
    """
    return repo_get_golden_answer(question_id)


def create_and_save_golden_answer(
    question,
    expert_answer_1,
    expert_answer_2,
    question_id=None,
):
    """
    Generate a Golden Answer using Qwen and save it to MongoDB.

    Workflow:
        Expert Answers
            ↓
        Golden Answer Agent
            ↓
        Structured Golden Answer
            ↓
        MongoDB
    """

    # Step 1: Generate Golden Answer using Qwen
    result = generate_golden_answer(
        expert_answer_1,
        expert_answer_2
    )

    golden_answer = result["golden_answer"]
    key_points = result["key_points"]

    # Step 2: Save generated result to MongoDB
    document_id = save_golden_answer(
        question=question,
        expert_answer_1=expert_answer_1,
        expert_answer_2=expert_answer_2,
        golden_answer=golden_answer,
        key_points=key_points,
        question_id=question_id,
    )

    # Step 3: Return complete result
    response = {
        "document_id": document_id,
        "question": question,
        "golden_answer": golden_answer,
        "key_points": key_points
    }
    if question_id is not None:
        response["question_id"] = question_id

    return response
