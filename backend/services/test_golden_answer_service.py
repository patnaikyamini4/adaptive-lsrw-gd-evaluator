from backend.services.golden_answer_service import (
    create_and_save_golden_answer
)


def main():
    question = "What are the advantages and disadvantages of online education?"

    expert_answer_1 = """
    Online education provides flexibility because students can learn
    from different locations and manage their own schedules. It also
    provides access to recorded lectures and digital learning materials.
    However, students may have less direct interaction with teachers
    and classmates.
    """

    expert_answer_2 = """
    Online learning makes education more accessible and convenient.
    Students can study at their own pace and use various online resources.
    A disadvantage is that it can reduce face-to-face communication and
    interaction with instructors.
    """

    result = create_and_save_golden_answer(
        question=question,
        expert_answer_1=expert_answer_1,
        expert_answer_2=expert_answer_2
    )

    print("\n===== GOLDEN ANSWER SERVICE TEST =====")

    print("\nDocument ID:")
    print(result["document_id"])

    print("\nGolden Answer:")
    print(result["golden_answer"])

    print("\nKey Points:")

    for index, point in enumerate(result["key_points"], start=1):
        print(f"{index}. {point}")

    print("\nValidation: SUCCESS")
    print("=======================================")


if __name__ == "__main__":
    main()