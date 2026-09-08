"""
Shared Automatic Speech Recognition Service.

Whisper is used for transcription.

This service does NOT identify speakers.

Speaker identity comes from:
    session_id
    participant_id

For GD, VAD segments can be supplied so that
Whisper processes only detected speech regions.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisper


class ASRService:
    """
    Shared Whisper ASR service.
    """

    def __init__(
        self,
        model_name: str = "base",
    ) -> None:

        self.model_name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading Whisper model: "
            f"{self.model_name}"
        )

        print(
            f"ASR device: "
            f"{self.device}"
        )

        self.model = whisper.load_model(
            self.model_name,
            device=self.device,
        )

        print(
            "Whisper model loaded successfully."
        )

    # ======================================================
    # FULL AUDIO TRANSCRIPTION
    # ======================================================

    def transcribe(
        self,
        audio_path: str,
        language: str | None = "en",
    ) -> dict[str, Any]:
        """
        Transcribe the complete audio file.

        This method is useful for generic ASR use.

        GD participant processing should preferably use
        transcribe_segments() after VAD.
        """

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        print()
        print("=" * 60)
        print("ASR")
        print("=" * 60)

        print(
            f"Audio: {path}"
        )

        print(
            f"Model: {self.model_name}"
        )

        print(
            f"Device: {self.device}"
        )

        result = self.model.transcribe(
            str(path),
            language=language,
            fp16=(
                self.device == "cuda"
            ),
            verbose=False,

            # Helps reduce repetitive hallucination.
            condition_on_previous_text=False,

            # Start with deterministic decoding.
            temperature=0.0,

            # Suppress very unlikely non-speech output.
            no_speech_threshold=0.6,
        )

        transcript_segments = []

        for segment in result.get(
            "segments",
            [],
        ):

            start = float(
                segment["start"]
            )

            end = float(
                segment["end"]
            )

            text = str(
                segment.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            transcript_segments.append(
                {
                    "start": round(
                        start,
                        3,
                    ),
                    "end": round(
                        end,
                        3,
                    ),
                    "duration": round(
                        end - start,
                        3,
                    ),
                    "text": text,
                }
            )

        return {
            "text": str(
                result.get(
                    "text",
                    "",
                )
            ).strip(),

            "segments": transcript_segments,

            "language": result.get(
                "language"
            ),

            "model": self.model_name,

            "device": self.device,
        }

    # ======================================================
    # VAD-AWARE TRANSCRIPTION
    # ======================================================

    def transcribe_segments(
        self,
        audio_path: str,
        speech_segments: list[dict[str, Any]],
        language: str | None = "en",
    ) -> dict[str, Any]:
        """
        Transcribe only the speech regions detected by VAD.

        Example VAD input:

            [
                {
                    "start": 0.8,
                    "end": 4.4,
                    "duration": 3.6
                },
                {
                    "start": 4.7,
                    "end": 13.1,
                    "duration": 8.4
                }
            ]

        Whisper processes each speech segment independently.

        This reduces hallucination caused by silence/non-speech
        regions and preserves the VAD timing information.
        """

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        if not speech_segments:
            return {
                "text": "",
                "segments": [],
                "language": language,
                "model": self.model_name,
                "device": self.device,
            }

        print()
        print("=" * 60)
        print("VAD-AWARE ASR")
        print("=" * 60)

        print(
            f"Audio: {path}"
        )

        print(
            f"Speech regions: "
            f"{len(speech_segments)}"
        )

        print(
            f"Model: {self.model_name}"
        )

        print(
            f"Device: {self.device}"
        )

        # Whisper expects 16 kHz audio.
        audio = whisper.load_audio(
            str(path)
        )

        sample_rate = 16000

        transcript_segments = []

        for index, speech_segment in enumerate(
            speech_segments,
            start=1,
        ):

            start = float(
                speech_segment["start"]
            )

            end = float(
                speech_segment["end"]
            )

            if end <= start:
                continue

            print()
            print(
                f"Speech segment {index}: "
                f"{start:.3f}s -> "
                f"{end:.3f}s"
            )

            start_sample = int(
                start * sample_rate
            )

            end_sample = int(
                end * sample_rate
            )

            # Keep the requested audio range
            # inside the actual audio.
            start_sample = max(
                0,
                start_sample,
            )

            end_sample = min(
                len(audio),
                end_sample,
            )

            segment_audio = audio[
                start_sample:end_sample
            ]

            if len(segment_audio) == 0:
                continue

            # Avoid sending extremely tiny pieces
            # to Whisper.
            duration = (
                len(segment_audio)
                / sample_rate
            )

            if duration < 0.25:
                continue

            result = self.model.transcribe(
                segment_audio,
                language=language,
                fp16=(
                    self.device == "cuda"
                ),
                verbose=False,

                # Important for short independent segments.
                condition_on_previous_text=False,

                temperature=0.0,

                no_speech_threshold=0.6,
            )

            text = str(
                result.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            # Whisper timestamps are relative
            # to this extracted speech segment.
            whisper_segments = result.get(
                "segments",
                [],
            )

            if whisper_segments:

                for whisper_segment in (
                    whisper_segments
                ):

                    relative_start = float(
                        whisper_segment["start"]
                    )

                    relative_end = float(
                        whisper_segment["end"]
                    )

                    segment_text = str(
                        whisper_segment.get(
                            "text",
                            "",
                        )
                    ).strip()

                    if not segment_text:
                        continue

                    absolute_start = (
                        start
                        + relative_start
                    )

                    absolute_end = (
                        start
                        + relative_end
                    )

                    transcript_segments.append(
                        {
                            "start": round(
                                absolute_start,
                                3,
                            ),
                            "end": round(
                                absolute_end,
                                3,
                            ),
                            "duration": round(
                                absolute_end
                                - absolute_start,
                                3,
                            ),
                            "text": segment_text,
                        }
                    )

            else:

                # Fallback if Whisper returns text
                # but no internal segments.
                transcript_segments.append(
                    {
                        "start": round(
                            start,
                            3,
                        ),
                        "end": round(
                            end,
                            3,
                        ),
                        "duration": round(
                            end - start,
                            3,
                        ),
                        "text": text,
                    }
                )

        transcript_segments.sort(
            key=lambda x: x["start"]
        )

        full_text = " ".join(
            segment["text"]
            for segment in transcript_segments
        )

        return {
            "text": full_text.strip(),

            "segments": transcript_segments,

            "language": language,

            "model": self.model_name,

            "device": self.device,
        }