from unittest.mock import Mock, patch

import pytest

from backend.ai.qwen_service import (
    MODEL_NAME,
    OPENROUTER_BASE_URL,
    ask_qwen,
)


def test_ask_qwen_sends_prompt_to_openrouter():
    fake_response = Mock()
    fake_response.choices = [
        Mock(
            message=Mock(
                content="Communication skills are important for students."
            )
        )
    ]

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "backend.ai.qwen_service._get_client",
        return_value=fake_client,
    ):
        result = ask_qwen(
            "Why are communication skills important?",
            max_tokens=200,
        )

    assert result == (
        "Communication skills are important for students."
    )

    fake_client.chat.completions.create.assert_called_once_with(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Why are communication skills important?",
            }
        ],
        max_tokens=200,
    )


def test_ask_qwen_rejects_empty_prompt():
    with pytest.raises(ValueError, match="non-empty"):
        ask_qwen("")


def test_ask_qwen_rejects_invalid_max_tokens():
    with pytest.raises(ValueError, match="integer"):
        ask_qwen("Hello", max_tokens="200")


def test_ask_qwen_rejects_non_positive_max_tokens():
    with pytest.raises(ValueError, match="greater than 0"):
        ask_qwen("Hello", max_tokens=0)


def test_qwen_constants():
    assert MODEL_NAME == "qwen/qwen3.8-max"
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"