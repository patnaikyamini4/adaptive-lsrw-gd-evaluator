"""
GD Voice Activity Detection Service

IMPORTANT ARCHITECTURE:

The application already knows WHO owns the audio.

Example:

    session_id = GD001
    participant_id = P001
    audio = P001 microphone stream

Therefore this module does NOT perform speaker diarization.

It only answers:

    "When was this participant speaking?"

Pipeline:

    Participant Audio
          ↓
        VAD
          ↓
    Speech Segments
          ↓
    ASR
          ↓
    Participant Transcript
"""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
)


class VADService:

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 80,
    ):
        """
        Initialize the VAD service.

        Parameters
        ----------
        threshold:
            Speech probability threshold.

        min_speech_duration_ms:
            Minimum duration required for a speech segment.

        min_silence_duration_ms:
            Silence shorter than this can be merged into
            surrounding speech.

        speech_pad_ms:
            Extra audio retained around detected speech.
        """

        self.threshold = threshold

        self.min_speech_duration_ms = (
            min_speech_duration_ms
        )

        self.min_silence_duration_ms = (
            min_silence_duration_ms
        )

        self.speech_pad_ms = speech_pad_ms

        # Silero VAD is lightweight and CPU-friendly.
        # We do not need to consume the RTX 3050 for this.
        self.device = torch.device("cpu")

        print(
            "Loading Silero VAD..."
        )

        self.model = load_silero_vad()

        self.model.eval()

        print(
            "Silero VAD loaded successfully."
        )

    # ========================================================
    # AUDIO LOADING
    # ========================================================

    @staticmethod
    def load_audio(
        audio_path: str,
    ) -> tuple[np.ndarray, int]:

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        waveform, sample_rate = sf.read(
            str(path),
            dtype="float32",
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        # Stereo / multi-channel → mono
        if waveform.ndim == 2:
            waveform = np.mean(
                waveform,
                axis=1,
            )

        if waveform.ndim != 1:
            raise ValueError(
                f"Unsupported audio shape: "
                f"{waveform.shape}"
            )

        return waveform, sample_rate

    # ========================================================
    # RESAMPLE TO 16 KHZ
    # ========================================================

    @staticmethod
    def resample_to_16khz(
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:

        if sample_rate == 16000:
            return waveform, sample_rate

        try:

            import torch
            import torchaudio.functional as F

            audio_tensor = torch.from_numpy(
                waveform
            ).float()

            resampled = F.resample(
                audio_tensor,
                orig_freq=sample_rate,
                new_freq=16000,
            )

            return (
                resampled.numpy(),
                16000,
            )

        except Exception as exc:

            raise RuntimeError(
                "Audio must be 16 kHz for GD VAD "
                "and automatic resampling failed. "
                f"Original sample rate: {sample_rate}. "
                f"Error: {exc}"
            ) from exc

    # ========================================================
    # DETECT SPEECH
    # ========================================================

    def detect(
        self,
        audio_path: str,
        session_id: str | None = None,
        participant_id: str | None = None,
    ) -> dict[str, Any]:

        waveform, sample_rate = (
            self.load_audio(audio_path)
        )

        original_duration = (
            len(waveform) / sample_rate
        )

        waveform, sample_rate = (
            self.resample_to_16khz(
                waveform,
                sample_rate,
            )
        )

        audio_tensor = torch.from_numpy(
            waveform
        ).float()

        # Silero expects one-dimensional audio.
        if audio_tensor.ndim != 1:
            audio_tensor = audio_tensor.squeeze()

        print()
        print("=" * 60)
        print("GD VAD")
        print("=" * 60)

        print(
            f"Session ID: {session_id}"
        )

        print(
            f"Participant ID: {participant_id}"
        )

        print(
            f"Original sample rate: "
            f"{sample_rate} Hz"
        )

        print(
            f"Original duration: "
            f"{original_duration:.3f}s"
        )

        print()
        print(
            "Running speech detection..."
        )

        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=(
                self.min_speech_duration_ms
            ),
            min_silence_duration_ms=(
                self.min_silence_duration_ms
            ),
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )

        segments = []

        for timestamp in timestamps:

            start = float(
                timestamp["start"]
            )

            end = float(
                timestamp["end"]
            )

            duration = end - start

            segments.append(
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
                        duration,
                        3,
                    ),
                }
            )

        total_speech_time = sum(
            segment["duration"]
            for segment in segments
        )

        return {
            "session_id": session_id,
            "participant_id": participant_id,
            "sample_rate": sample_rate,
            "audio_duration": round(
                original_duration,
                3,
            ),
            "speech_duration": round(
                total_speech_time,
                3,
            ),
            "speech_ratio": round(
                total_speech_time
                / original_duration
                if original_duration > 0
                else 0.0,
                4,
            ),
            "segments": segments,
        }

    # ========================================================
    # DETECT FROM IN-MEMORY AUDIO
    # ========================================================

    def detect_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        session_id: str | None = None,
        participant_id: str | None = None,
    ) -> dict[str, Any]:

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.ndim == 2:

            waveform = np.mean(
                waveform,
                axis=1,
            )

        if waveform.ndim != 1:
            raise ValueError(
                "Waveform must be mono."
            )

        duration = (
            len(waveform)
            / sample_rate
        )

        waveform, sample_rate = (
            self.resample_to_16khz(
                waveform,
                sample_rate,
            )
        )

        audio_tensor = torch.from_numpy(
            waveform
        ).float()

        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=(
                self.min_speech_duration_ms
            ),
            min_silence_duration_ms=(
                self.min_silence_duration_ms
            ),
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )

        segments = []

        for timestamp in timestamps:

            start = float(
                timestamp["start"]
            )

            end = float(
                timestamp["end"]
            )

            segments.append(
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
                }
            )

        total_speech_time = sum(
            segment["duration"]
            for segment in segments
        )

        return {
            "session_id": session_id,
            "participant_id": participant_id,
            "sample_rate": sample_rate,
            "audio_duration": round(
                duration,
                3,
            ),
            "speech_duration": round(
                total_speech_time,
                3,
            ),
            "speech_ratio": round(
                total_speech_time / duration
                if duration > 0
                else 0.0,
                4,
            ),
            "segments": segments,
        }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ADAPTIVE LSRW GD - VAD TEST")
    print("=" * 60)

    audio_path = (
        Path("data")
        / "gd"
        / "audio"
        / "p001_test.wav"
    )

    if not audio_path.exists():

        print()
        print(
            "No test audio found."
        )

        print()
        print(
            "This is expected because the old "
            "diarization test audio was deleted."
        )

        print()
        print(
            "VADService itself is ready."
        )

    else:

        service = VADService()

        result = service.detect(
            audio_path=str(audio_path),
            session_id="GD001",
            participant_id="P001",
        )

        print()
        print("=" * 60)
        print("VAD RESULT")
        print("=" * 60)

        print(
            f"Session: "
            f"{result['session_id']}"
        )

        print(
            f"Participant: "
            f"{result['participant_id']}"
        )

        print(
            f"Audio duration: "
            f"{result['audio_duration']:.3f}s"
        )

        print(
            f"Speech duration: "
            f"{result['speech_duration']:.3f}s"
        )

        print(
            f"Speech ratio: "
            f"{result['speech_ratio']:.2%}"
        )

        print(
            f"Speech segments: "
            f"{len(result['segments'])}"
        )

        print()

        for index, segment in enumerate(
            result["segments"],
            start=1,
        ):

            print(
                f"{index:03d}. "
                f"{segment['start']:.3f}s -> "
                f"{segment['end']:.3f}s "
                f"({segment['duration']:.3f}s)"
            )