from backend.services.golden_answer_repository import save_golden_answer


question = "What are the advantages and disadvantages of online education?"

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

golden_answer = """
Online education makes learning more flexible and accessible
by allowing students to study from different locations, manage
their own schedules, and use digital resources such as recorded
lectures. However, it may reduce direct interaction between
students and teachers.
"""

key_points = [
    "Online education offers flexibility in location and scheduling",
    "It increases access to educational resources",
    "It may reduce direct interaction between students and teachers"
]


try:
    document_id = save_golden_answer(
        question,
        expert_answer_1,
        expert_answer_2,
        golden_answer,
        key_points
    )

    print("\n===== MONGODB SAVE TEST =====")
    print("Golden Answer saved successfully")
    print("Document ID:", document_id)
    print("=============================\n")

except Exception as e:
    print("\n===== MONGODB SAVE ERROR =====")
    print(type(e).__name__, ":", e)
    print("==============================\n")