from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading semantic embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Semantic embedding model loaded successfully!")
    print(f"Model: {MODEL_NAME}")

    # Test sentence
    sentence = "Online education provides flexibility for students."

    embedding = model.encode(sentence)

    print(f"Embedding type: {type(embedding)}")
    print(f"Embedding dimensions: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    print("\nValidation: SUCCESS")


if __name__ == "__main__":
    main()