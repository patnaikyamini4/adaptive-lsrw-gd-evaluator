from backend.ai.gd.agents.base_agent import GDBaseAgent
from backend.ai.gd.gd_agent_input import GDParticipantInput


class RelevanceAgent(GDBaseAgent):

    agent_name = "relevance"

    def build_prompt(
        self,
        participant_input: GDParticipantInput
    ) -> str:

        return f"""
You are the Relevance Evaluation Agent for an
English Group Discussion assessment system.

Evaluate ONLY the relevance of participant
{participant_input.participant_id}'s contribution.

GD Topic:
{participant_input.topic}

Participant Transcript:
{participant_input.participant_transcript}

Full Group Discussion Transcript:
{participant_input.group_transcript}

Evaluate whether the participant:

1. Directly addresses the GD topic.
2. Stays focused on the discussion.
3. Provides ideas related to the topic.
4. Connects contributions to the ongoing discussion.
5. Avoids irrelevant or unrelated content.

Do NOT evaluate:

- speaking duration
- speaking speed
- grammar
- pronunciation
- number of turns
- participation frequency

Those are evaluated by other components.

Important:

Evaluate the participant's actual contribution.
Do not invent statements that are not present
in the transcript.

Because the transcript was produced by speech
recognition, minor transcription errors may exist.
Do not penalize the participant simply because
one unusual word appears to be an ASR error.

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