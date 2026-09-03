from backend.ai.golden_answer_agent import generate_golden_answer


expert_answer_1 = """
Online education provides flexibility because students can
learn from different locations and manage their own schedules.
It also provides access to educational resources for students
who cannot attend traditional classrooms.
"""


expert_answer_2 = """
Online learning makes education more accessible and convenient.
Students can study remotely and often have access to recorded
lectures and digital learning materials. However, it may reduce
direct interaction between students and teachers.
"""


try:
    result = generate_golden_answer(
        expert_answer_1,
        expert_answer_2
    )

    print("\n===== STRUCTURED GOLDEN ANSWER =====")

    print("Golden Answer:")
    print(result["golden_answer"])

    print("\nKey Points:")

    for index, point in enumerate(result["key_points"], start=1):
        print(f"{index}. {point}")

    print("\nData Type:", type(result))
    print("Validation: SUCCESS")

    print("====================================\n")

except Exception as e:
    print("\n===== GOLDEN ANSWER ERROR =====")
    print(type(e).__name__, ":", e)
    print("================================\n")