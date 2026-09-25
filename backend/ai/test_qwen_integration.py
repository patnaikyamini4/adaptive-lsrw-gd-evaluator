from backend.ai.qwen_service import ask_qwen


def main():
    print("Testing shared Qwen service...")

    prompt = """
You are an English language assessment assistant.

Explain in 2 sentences why communication skills
are important for a college student.

Return only the answer.
"""

    response = ask_qwen(prompt, max_tokens=200)

    print("\n===== QWEN RESPONSE =====")
    print(response)
    print("\nQwen integration test: PASSED")


if __name__ == "__main__":
    main()