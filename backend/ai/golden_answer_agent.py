import json

from backend.ai.qwen_service import ask_qwen


def generate_golden_answer(expert_answer_1, expert_answer_2):
    """
    Generate a structured Golden Answer from two expert answers.

    Returns:
        dict containing:
        - golden_answer
        - key_points
    """

    prompt = f"""
You are the Golden Answer Agent for an adaptive English
language assessment system.

Two experts have provided answers to the same question.

Expert Answer 1:
{expert_answer_1}

Expert Answer 2:
{expert_answer_2}

Create one high-quality Golden Answer by combining the
important ideas from both expert answers.

Requirements:
1. Preserve important ideas from both experts.
2. Remove unnecessary repetition.
3. Do not invent unrelated information.
4. Use clear and natural English.
5. The answer should be suitable as a reference answer
   for evaluating candidate responses.
6. Identify the important ideas that a candidate response
   should ideally cover.
7. Return ONLY valid JSON.
8. Do not use Markdown.
9. Do not add explanations outside the JSON.

Use exactly this JSON structure:

{{
    "golden_answer": "A complete reference answer here",
    "key_points": [
        "Important point 1",
        "Important point 2",
        "Important point 3"
    ]
}}
"""

    response_text = ask_qwen(prompt, max_tokens=1500)

    # Remove accidental Markdown code fences if the model adds them
    response_text = response_text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]

    if response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    # Convert JSON text into a Python dictionary
    result = json.loads(response_text)

    # Validate required fields
    if "golden_answer" not in result:
        raise ValueError("Missing 'golden_answer' field")

    if "key_points" not in result:
        raise ValueError("Missing 'key_points' field")

    # Validate field types
    if not isinstance(result["golden_answer"], str):
        raise ValueError("'golden_answer' must be a string")

    if not isinstance(result["key_points"], list):
        raise ValueError("'key_points' must be a list")

    if len(result["key_points"]) == 0:
        raise ValueError("'key_points' cannot be empty")

    return result