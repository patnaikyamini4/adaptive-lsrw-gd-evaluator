from backend.ai.gd.agents.base_agent import GDBaseAgent
from backend.ai.gd.gd_agent_input import GDParticipantInput


class ParticipationAgent(GDBaseAgent):

    agent_name = "participation"

    def build_prompt(
        self,
        participant_input: GDParticipantInput
    ) -> str:

        interaction = participant_input.interaction_features

        return f"""
You are the Participation Evaluation Agent for an
English Group Discussion assessment system.

Evaluate the quality of participant
{participant_input.participant_id}'s participation.

GD Topic:
{participant_input.topic}

Participant Transcript:
{participant_input.participant_transcript}

Full Group Discussion Transcript:
{participant_input.group_transcript}

Objective interaction features:

First speaking time:
{interaction.get("first_speaking_time", 0)}

Last speaking time:
{interaction.get("last_speaking_time", 0)}

Responses:
{interaction.get("responses", 0)}

Overlap events:
{interaction.get("overlap_events", 0)}

Other speakers before:
{interaction.get("other_speakers_before", 0)}

Other speakers after:
{interaction.get("other_speakers_after", 0)}

Average gap between turns:
{interaction.get("average_gap_between_turns", 0)}

Evaluate:

1. Whether the participant contributes meaningfully.
2. Whether the participant responds to other speakers.
3. Whether the participant builds on previous ideas.
4. Whether the participant introduces useful points.
5. Whether participation is balanced.
6. Whether interaction with other participants is constructive.
7. Whether the participant is excessively passive or dominant.

Important:

An overlap event is ONLY an objective temporal overlap.

Do NOT automatically treat every overlap as a negative
interruption.

Consider the surrounding transcript to determine
whether the interaction appears constructive,
accidental, or disruptive.

A response count is only a timing-based heuristic.

Use the actual transcript to determine whether the
participant meaningfully responds to another speaker.

Do NOT evaluate grammar as the primary criterion.

Do NOT judge participation only by speaking time.

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
        "Specific evidence from the discussion"
    ]
}}

Score must be between 0 and 100.
"""