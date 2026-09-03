from qwen_service import ask_qwen


prompt = """
You are an English language assessment assistant.

Explain in 2 sentences why communication skills are important
for a college student.
"""

try:
    result = ask_qwen(prompt)

    print("\n===== QWEN RESPONSE =====")
    print(result)
    print("=========================\n")

except Exception as e:
    print("\n===== QWEN ERROR =====")
    print(e)
    print("======================\n")