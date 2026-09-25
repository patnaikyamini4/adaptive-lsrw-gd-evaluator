"""
Base class for GD evaluation agents.

All GD agents use the shared Qwen service.

The base class is responsible for:

1. Building an agent prompt
2. Calling the shared LLM
3. Parsing JSON safely
4. Validating the response
5. Returning GDAgentResult
"""

from abc import ABC, abstractmethod
import json
import re
from typing import Any

from backend.ai.gd.gd_agent_input import (
    GDParticipantInput,
)

from backend.ai.gd.gd_agent_result import (
    GDAgentResult,
)


class GDBaseAgent(ABC):

    # ======================================================
    # AGENT NAME
    # ======================================================

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """
        Return the agent name.
        """
        raise NotImplementedError

    # ======================================================
    # PROMPT
    # ======================================================

    @abstractmethod
    def build_prompt(
        self,
        input_data: GDParticipantInput,
    ) -> str:
        """
        Build the prompt sent to the LLM.
        """
        raise NotImplementedError

    # ======================================================
    # EVALUATE
    # ======================================================

    def evaluate(
        self,
        input_data: GDParticipantInput,
    ) -> GDAgentResult:

        prompt = self.build_prompt(
            input_data
        )

        response_text = self.call_llm(
            prompt
        )

        data = self.parse_response(
            response_text
        )

        return GDAgentResult(

            participant_id=(
                input_data.participant_id
            ),

            agent=self.agent_name,

            score=data["score"],

            reasoning=data.get(
                "reasoning",
                "",
            ),

            strengths=data.get(
                "strengths",
                [],
            ),

            weaknesses=data.get(
                "weaknesses",
                [],
            ),

            evidence=data.get(
                "evidence",
                [],
            ),

            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # ======================================================
    # SHARED LLM
    # ======================================================

    def call_llm(
        self,
        prompt: str,
    ) -> str:

        from backend.ai.qwen_service import (
            ask_qwen,
        )

        return ask_qwen(
            prompt,
            max_tokens=1200,
        )

    # ======================================================
    # SAFE JSON PARSING
    # ======================================================

    def parse_response(
        self,
        response_text: str,
    ) -> dict[str, Any]:

        if not response_text:
            raise ValueError(
                f"{self.agent_name} agent returned an empty response"
            )

        raw = response_text.strip()

        # --------------------------------------------------
        # Debug information
        # --------------------------------------------------

        print()
        print(
            f"[{self.agent_name}] Raw LLM response:"
        )
        print("-" * 70)
        print(raw)
        print("-" * 70)

        # --------------------------------------------------
        # Remove Markdown code fences
        # --------------------------------------------------

        cleaned = self._remove_code_fences(
            raw
        )

        # --------------------------------------------------
        # First JSON attempt
        # --------------------------------------------------

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            # ------------------------------------------------
            # Second attempt:
            # Extract the outermost JSON object.
            # ------------------------------------------------

            extracted = (
                self._extract_json_object(
                    cleaned
                )
            )

            if extracted is None:

                raise ValueError(
                    f"{self.agent_name} agent returned invalid JSON:\n"
                    f"{cleaned}"
                )

            try:

                data = json.loads(
                    extracted
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"{self.agent_name} agent returned malformed JSON: "
                    f"{exc}\n\n"
                    f"LLM response:\n"
                    f"{cleaned}"
                ) from exc

        # --------------------------------------------------
        # Validate top-level object
        # --------------------------------------------------

        if not isinstance(data, dict):

            raise ValueError(
                f"{self.agent_name} agent response "
                f"must be a JSON object"
            )

        # --------------------------------------------------
        # Required score
        # --------------------------------------------------

        if "score" not in data:

            raise ValueError(
                f"{self.agent_name} agent response "
                f"is missing 'score'"
            )

        score = data["score"]

        # --------------------------------------------------
        # Convert numeric score
        # --------------------------------------------------

        try:

            score = float(score)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"{self.agent_name} agent score "
                f"must be numeric"
            ) from exc

        # --------------------------------------------------
        # Score range
        # --------------------------------------------------

        if not 0.0 <= score <= 100.0:

            raise ValueError(
                f"{self.agent_name} agent score "
                f"must be between 0 and 100"
            )

        data["score"] = score

        # --------------------------------------------------
        # Reasoning
        # --------------------------------------------------

        reasoning = data.get(
            "reasoning",
            "",
        )

        if not isinstance(
            reasoning,
            str,
        ):

            reasoning = str(
                reasoning
            )

        data["reasoning"] = reasoning

        # --------------------------------------------------
        # Strengths
        # --------------------------------------------------

        data["strengths"] = (
            self._ensure_string_list(
                data.get(
                    "strengths",
                    [],
                )
            )
        )

        # --------------------------------------------------
        # Weaknesses
        # --------------------------------------------------

        data["weaknesses"] = (
            self._ensure_string_list(
                data.get(
                    "weaknesses",
                    [],
                )
            )
        )

        # --------------------------------------------------
        # Evidence
        # --------------------------------------------------

        data["evidence"] = (
            self._ensure_string_list(
                data.get(
                    "evidence",
                    [],
                )
            )
        )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        data["metadata"] = metadata

        return data

    # ======================================================
    # REMOVE CODE FENCES
    # ======================================================

    @staticmethod
    def _remove_code_fences(
        text: str,
    ) -> str:

        text = text.strip()

        if text.startswith(
            "```json"
        ):

            text = text[
                len("```json"):
            ]

        elif text.startswith(
            "```"
        ):

            text = text[
                len("```"):
            ]

        if text.endswith(
            "```"
        ):

            text = text[
                :-len("```")
            ]

        return text.strip()

    # ======================================================
    # EXTRACT JSON OBJECT
    # ======================================================

    @staticmethod
    def _extract_json_object(
        text: str,
    ) -> str | None:

        start = text.find("{")

        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for index in range(
            start,
            len(text),
        ):

            char = text[index]

            # ----------------------------------------------
            # Handle escaped characters inside strings
            # ----------------------------------------------

            if escape:

                escape = False
                continue

            if char == "\\" and in_string:

                escape = True
                continue

            # ----------------------------------------------
            # Handle strings
            # ----------------------------------------------

            if char == '"':

                in_string = not in_string
                continue

            # ----------------------------------------------
            # Ignore braces inside strings
            # ----------------------------------------------

            if in_string:
                continue

            # ----------------------------------------------
            # Track JSON object depth
            # ----------------------------------------------

            if char == "{":

                depth += 1

            elif char == "}":

                depth -= 1

                if depth == 0:

                    return text[
                        start:index + 1
                    ]

        return None

    # ======================================================
    # STRING LIST VALIDATION
    # ======================================================

    @staticmethod
    def _ensure_string_list(
        value: Any,
    ) -> list[str]:

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):

            return [value]

        if not isinstance(
            value,
            list,
        ):

            return [
                str(value)
            ]

        return [
            str(item)
            for item in value
        ]