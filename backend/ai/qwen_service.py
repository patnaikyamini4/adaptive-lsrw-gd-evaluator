import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from the project .env file.
load_dotenv()


MODEL_NAME = "qwen/qwen3.8-max"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_api_key() -> str:
    """
    Read the OpenRouter API key only when a Qwen request is made.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in the .env file"
        )

    return api_key


def _get_client() -> OpenAI:
    """
    Create the shared OpenRouter OpenAI-compatible client.
    """
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_get_api_key(),
        timeout=60.0,
        max_retries=0,
    )


def ask_qwen(prompt: str, max_tokens: int = 1000) -> str:
    """
    Send one prompt to Qwen through OpenRouter.

    Qwen is used on-demand. It is not called continuously during
    the live GD session.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise RuntimeError("Qwen returned no choices")

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("Qwen returned an empty response")

    return content.strip()