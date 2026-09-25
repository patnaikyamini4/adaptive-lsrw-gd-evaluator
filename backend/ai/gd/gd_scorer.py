import math
from typing import Any

from backend.ai.gd.gd_agent_result import GDAgentResult


class GDScorer:

    DEFAULT_WEIGHTS = {
        "relevance": 0.25,
        "coherence": 0.25,
        "fluency": 0.25,
        "participation": 0.25,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None
    ):
        self.weights = (
            weights.copy()
            if weights is not None
            else self.DEFAULT_WEIGHTS.copy()
        )

        self._validate_weights()

    def _validate_weights(self) -> None:

        expected_agents = set(self.DEFAULT_WEIGHTS)

        if set(self.weights) != expected_agents:
            raise ValueError(
                "Weights must contain exactly: "
                "relevance, coherence, fluency, participation"
            )

        for agent, weight in self.weights.items():

            if isinstance(weight, bool) or not isinstance(
                weight,
                (int, float),
            ):
                raise ValueError(
                    f"Weight for {agent} must be numeric"
                )

            if not math.isfinite(float(weight)):
                raise ValueError(
                    f"Weight for {agent} must be finite"
                )

            if weight < 0:
                raise ValueError(
                    f"Weight for {agent} cannot be negative"
                )

        total = sum(self.weights.values())

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Agent weights must sum to 1.0, got {total}"
            )

    def calculate(
        self,
        results: list[GDAgentResult]
    ) -> dict[str, Any]:

        if not results:
            raise ValueError(
                "No agent results provided"
            )

        expected_agents = set(self.weights)
        agent_names = [result.agent for result in results]

        duplicate_agents = sorted(
            {
                agent
                for agent in agent_names
                if agent_names.count(agent) > 1
            }
        )

        if duplicate_agents:
            raise ValueError(
                f"Duplicate agent results: {duplicate_agents}"
            )

        received_agents = set(agent_names)

        unexpected = received_agents - expected_agents

        if unexpected:
            raise ValueError(
                f"Unexpected agent results: {sorted(unexpected)}"
            )

        missing = (
            expected_agents
            - received_agents
        )

        if missing:
            raise ValueError(
                f"Missing agent results: {sorted(missing)}"
            )

        scores: dict[str, float] = {}

        for result in results:

            score = result.score

            if isinstance(score, bool) or not isinstance(
                score,
                (int, float),
            ):
                raise ValueError(
                    f"Score for {result.agent} must be numeric"
                )

            score = float(score)

            if not math.isfinite(score):
                raise ValueError(
                    f"Score for {result.agent} must be finite"
                )

            if not 0.0 <= score <= 100.0:
                raise ValueError(
                    f"Score for {result.agent} must be between 0 and 100"
                )

            scores[result.agent] = score

        participant_ids = {
            result.participant_id
            for result in results
        }

        if len(participant_ids) != 1:
            raise ValueError(
                "All agent results must belong "
                "to the same participant"
            )

        participant_id = results[0].participant_id

        final_score = sum(
            scores[agent] * weight
            for agent, weight in self.weights.items()
        )

        if not math.isfinite(final_score):
            raise ValueError("Final score must be finite")

        if not 0.0 <= final_score <= 100.0:
            raise ValueError(
                "Final score must be between 0 and 100"
            )

        return {
            "participant_id": participant_id,
            "scores": scores,
            "weights": self.weights.copy(),
            "final_score": round(final_score, 2),
        }
