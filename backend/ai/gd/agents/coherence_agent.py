from backend.ai.gd.agents.base_agent import GDBaseAgent
from backend.ai.gd.gd_agent_input import GDParticipantInput


class CoherenceAgent(GDBaseAgent):

    agent_name = "coherence"

    def build_prompt(
        self,
        participant_input: GDParticipantInput
    ) -> str:

        return f"""
You are the Coherence Evaluation Agent for an
English Group Discussion assessment system.

Evaluate ONLY the coherence of participant
{participant_input.participant_id}'s contribution.

GD Topic:
{participant_input.topic}

Participant Transcript:
{participant_input.participant_transcript}

Full Group Discussion Transcript:
{participant_input.group_transcript}

Evaluate whether the participant:

1. Expresses ideas in a logical order.
2. Maintains a clear flow of thought.
3. Connects ideas appropriately.
4. Avoids contradictory or disconnected statements.
5. Develops ideas sufficiently for a group discussion.
6. Makes the overall contribution understandable.

Consider the participant's contributions as a whole.

Do NOT evaluate:

- speaking duration
- speaking speed
- pronunciation
- raw speaking time
- participation frequency
- number of turns

Those are evaluated elsewhere.

Important:

Evaluate the logical organization of the ideas,
not merely grammar correctness.

Because the transcript comes from speech recognition,
minor ASR errors may occur. Do not treat every
transcription error as a coherence problem.

Return ONLY valid JSON.
Do not use Markdown.

Use exactly this structure:

{{
    "score": 0,
    "reasoning": "Brief explanation",
    "strengths": [
        "Strength"
    ],
    "weaknesses": [
        "Weakness"
    ],
    "evidence": [
        "Specific evidence from the participant contribution"
    ]
}}

Score must be between 0 and 100.
"""