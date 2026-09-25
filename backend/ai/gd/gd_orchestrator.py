from typing import Any

from backend.ai.gd.agents import (
    CoherenceAgent,
    FluencyAgent,
    ParticipationAgent,
    RelevanceAgent,
)
from backend.ai.gd.gd_agent_input import GDParticipantInput
from backend.ai.gd.gd_scorer import GDScorer


class GDOrchestrator:

    def __init__(
        self,
        scorer: GDScorer | None = None
    ):

        self.relevance_agent = RelevanceAgent()
        self.coherence_agent = CoherenceAgent()
        self.fluency_agent = FluencyAgent()
        self.participation_agent = ParticipationAgent()

        self.scorer = scorer or GDScorer()

    def evaluate_participant(
        self,
        participant_input: GDParticipantInput
    ) -> dict[str, Any]:

        results = []

        # Relevance
        relevance_result = (
            self.relevance_agent.evaluate(
                participant_input
            )
        )

        results.append(relevance_result)

        # Coherence
        coherence_result = (
            self.coherence_agent.evaluate(
                participant_input
            )
        )

        results.append(coherence_result)

        # Fluency
        fluency_result = (
            self.fluency_agent.evaluate(
                participant_input
            )
        )

        results.append(fluency_result)

        # Participation
        participation_result = (
            self.participation_agent.evaluate(
                participant_input
            )
        )

        results.append(participation_result)

        # Calculate final score.
        scorecard = self.scorer.calculate(results)

        return {
            "session_id": participant_input.session_id,
            "participant_id": participant_input.participant_id,
            "topic": participant_input.topic,
            "agent_results": [
                result.to_dict()
                for result in results
            ],
            "scorecard": scorecard,
        }

    def evaluate_session(
        self,
        participants: list[GDParticipantInput]
    ) -> list[dict[str, Any]]:

        reports = []

        for participant in participants:

            report = self.evaluate_participant(
                participant
            )

            reports.append(report)

        return reports