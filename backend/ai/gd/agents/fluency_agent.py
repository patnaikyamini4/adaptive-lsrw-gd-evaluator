from backend.ai.gd.agents.base_agent import GDBaseAgent
from backend.ai.gd.gd_agent_input import GDParticipantInput


class FluencyAgent(GDBaseAgent):

    agent_name = "fluency"

    def build_prompt(
        self,
        participant_input: GDParticipantInput
    ) -> str:

        features = participant_input.features

        return f"""
You are the Fluency Evaluation Agent for an
English Group Discussion assessment system.

Evaluate the English fluency demonstrated in the
participant's transcript.

Participant:
{participant_input.participant_id}

GD Topic:
{participant_input.topic}

Participant Transcript:
{participant_input.participant_transcript}

Objective speech features:

Speaking time:
{features.get("speaking_time", 0)}

Speaking ratio:
{features.get("speaking_ratio", 0)}

Word count:
{features.get("word_count", 0)}

Words per minute:
{features.get("words_per_minute", 0)}

Turn count:
{features.get("turn_count", 0)}

Average turn duration:
{features.get("average_turn_duration", 0)}

Longest turn:
{features.get("longest_turn", 0)}

Shortest turn:
{features.get("shortest_turn", 0)}

Evaluate:

1. Natural flow of English.
2. Ability to express ideas continuously.
3. Sentence formation.
4. Appropriate vocabulary.
5. Clarity of expression.
6. Grammatical problems that affect communication.
7. Repetition or fragmented expression.

Important:

The objective timing values are supporting information.

Do NOT calculate the score using WPM alone.

Do NOT assume that faster speech means better fluency.

Do NOT assume that slower speech means poor fluency.

Speech-to-text transcription may contain recognition
errors. Do not treat every unusual word as a definite
candidate language error.

Evaluate the overall linguistic quality demonstrated
by the transcript.

Do NOT evaluate:

- topic relevance
- participation quality
- speaking time as the primary criterion

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
        "Specific transcript evidence"
    ]
}}

Score must be between 0 and 100.
"""