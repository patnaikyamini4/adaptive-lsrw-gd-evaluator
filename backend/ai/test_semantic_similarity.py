from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def calculate_similarity(model, sentence_a, sentence_b):
    """
    Calculate cosine similarity between two sentences.
    """

    embeddings = model.encode(
        [sentence_a, sentence_b]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


def main():

    print("Loading semantic embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.\n")

    golden_answer = (
        "Online education provides flexibility for students."
    )

    candidate_high = (
        "Students can study online from different locations "
        "with flexible schedules."
    )

    candidate_medium = (
        "Online learning provides recorded lectures and "
        "digital learning materials."
    )

    candidate_low = (
        "The weather is pleasant today."
    )

    # -----------------------------------------------------
    # Test 1
    # -----------------------------------------------------

    high_similarity = calculate_similarity(
        model,
        golden_answer,
        candidate_high
    )

    # -----------------------------------------------------
    # Test 2
    # -----------------------------------------------------

    medium_similarity = calculate_similarity(
        model,
        golden_answer,
        candidate_medium
    )

    # -----------------------------------------------------
    # Test 3
    # -----------------------------------------------------

    low_similarity = calculate_similarity(
        model,
        golden_answer,
        candidate_low
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("===== SEMANTIC SIMILARITY TEST =====")

    print("\nTest 1 - High Similarity")
    print(f"Golden   : {golden_answer}")
    print(f"Candidate: {candidate_high}")
    print(f"Similarity: {high_similarity:.4f}")

    print("\nTest 2 - Medium Similarity")
    print(f"Golden   : {golden_answer}")
    print(f"Candidate: {candidate_medium}")
    print(f"Similarity: {medium_similarity:.4f}")

    print("\nTest 3 - Low Similarity")
    print(f"Golden   : {golden_answer}")
    print(f"Candidate: {candidate_low}")
    print(f"Similarity: {low_similarity:.4f}")

    print("\n===== VALIDATION =====")

    if (
        high_similarity > medium_similarity
        and medium_similarity > low_similarity
    ):
        print("Validation: SUCCESS")
    else:
        print("Validation: CHECK RESULTS")

    print("=======================")


if __name__ == "__main__":
    main()