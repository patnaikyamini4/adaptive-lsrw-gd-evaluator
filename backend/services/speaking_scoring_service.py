"""
Speaking Scoring Service.

Computes a deterministic, transparent, and bounded (0-100) scoring breakdown
for candidate speaking responses.
"""

from typing import Any


class SpeakingScoringService:
    """
    Deterministic scoring engine for LSRW Speaking responses.

    Scoring Model (v1.0):
    1. Semantic Relevance (Weight: 1.0):
       - Uses cosine similarity s in [-1.0, 1.0] from sentence-transformers.
       - Bounded mapping:
           score = round(max(0.0, min(100.0, s * 100.0)), 2)
       - Values <= 0.0 map to 0.0 (unrelated or negative semantic alignment).
       - Values in [0.0, 1.0] scale linearly to [0.0, 100.0].
    2. Fluency / Delivery Proxy:
       - Documented as unmeasured in v1.0 because single-pass ASR transcript
         does not provide verified pause/prosody/timing alignment data.
       - No unverified metrics are fabricated.
    3. Final Score:
       - Weighted combination: final_score = semantic_score * 1.0
       - Bounded strictly within [0.0, 100.0].
    """

    WEIGHT_SEMANTIC = 1.0
    SCORING_METHOD = "deterministic_semantic_v1"
    VERSION = "1.0"

    def calculate_score(
        self,
        transcript: str,
        semantic_similarity: float,
        asr_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate deterministic speaking score breakdown.

        Args:
            transcript: Transcribed text of candidate audio.
            semantic_similarity: Cosine similarity float between golden answer and candidate transcript.
            asr_result: Optional ASR metadata dictionary.

        Returns:
            Dictionary containing final_score, component breakdown, scoring method, and version.
        """
        # Ensure semantic similarity is float
        raw_similarity = float(semantic_similarity) if semantic_similarity is not None else 0.0

        # Linear bounded mapping: s in [-1.0, 1.0] -> [0.0, 100.0]
        # s <= 0.0 -> 0.0, s in (0.0, 1.0] -> s * 100.0, s >= 1.0 -> 100.0
        semantic_score = round(max(0.0, min(100.0, raw_similarity * 100.0)), 2)

        final_score = round(semantic_score * self.WEIGHT_SEMANTIC, 2)
        # Ensure final_score is clamped strictly between 0.0 and 100.0
        final_score = max(0.0, min(100.0, final_score))

        return {
            "final_score": final_score,
            "components": {
                "semantic": {
                    "raw_similarity": raw_similarity,
                    "score": semantic_score,
                    "weight": self.WEIGHT_SEMANTIC,
                }
            },
            "scoring_method": self.SCORING_METHOD,
            "version": self.VERSION,
        }

