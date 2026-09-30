from unittest.mock import MagicMock, patch
from bson import ObjectId

from backend.services import golden_answer_repository
from backend.services import golden_answer_service


def test_repository_get_golden_answer_found():
    fake_doc = {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "question_id": "speaking-q1",
        "question": "Describe your hometown.",
        "golden_answer": "My hometown is known for its rich culture.",
        "key_points": ["culture", "history"],
    }

    mock_collection = MagicMock()
    mock_collection.find_one.return_value = fake_doc

    with patch.object(golden_answer_repository, "collection", mock_collection):
        result = golden_answer_repository.get_golden_answer("speaking-q1")

    assert result is not None
    assert result["question_id"] == "speaking-q1"
    assert result["golden_answer"] == "My hometown is known for its rich culture."
    assert "_id" not in result
    mock_collection.find_one.assert_called_once_with({"question_id": "speaking-q1"})


def test_repository_get_golden_answer_not_found():
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = None

    with patch.object(golden_answer_repository, "collection", mock_collection):
        result = golden_answer_repository.get_golden_answer("missing-question")

    assert result is None
    mock_collection.find_one.assert_called_once_with({"question_id": "missing-question"})


def test_repository_get_golden_answer_removes_id():
    fake_doc = {
        "_id": ObjectId("507f1f77bcf86cd799439011"),
        "question_id": "speaking-q2",
        "golden_answer": "Sample answer",
    }

    mock_collection = MagicMock()
    mock_collection.find_one.return_value = fake_doc

    with patch.object(golden_answer_repository, "collection", mock_collection):
        result = golden_answer_repository.get_golden_answer("speaking-q2")

    assert "_id" not in result
    assert result["golden_answer"] == "Sample answer"


def test_service_get_golden_answer_delegates_to_repository():
    expected_doc = {
        "question_id": "speaking-q3",
        "golden_answer": "Service layer delegation test answer.",
        "key_points": ["delegation"],
    }

    with patch.object(
        golden_answer_service,
        "repo_get_golden_answer",
        return_value=expected_doc,
    ) as mock_repo_get:
        result = golden_answer_service.get_golden_answer("speaking-q3")

    assert result == expected_doc
    mock_repo_get.assert_called_once_with("speaking-q3")

