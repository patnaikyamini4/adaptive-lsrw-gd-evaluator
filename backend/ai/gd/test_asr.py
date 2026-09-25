"""
GD Shared ASR Test

Tests the shared ASR service on P001 audio.

No speaker diarization is used.
"""


from unittest.mock import Mock

import backend.ai.asr_service as asr_service_module
from backend.ai.asr_service import ASRService


def test_transcribe_audio_uses_a_lazy_reusable_service(
    monkeypatch,
):
    service = Mock()
    service.transcribe.return_value = {
        "text": "Shared ASR transcript.",
        "segments": [
            {
                "start": 1.0,
                "end": 2.0,
                "duration": 1.0,
                "text": "Shared ASR transcript.",
            }
        ],
        "language": "en",
        "model": "base",
        "device": "cpu",
    }
    service_class = Mock(return_value=service)

    monkeypatch.setattr(
        asr_service_module,
        "_shared_asr_service",
        None,
    )
    monkeypatch.setattr(
        asr_service_module,
        "ASRService",
        service_class,
    )

    default_result = asr_service_module.transcribe_audio(
        "sample.wav"
    )
    explicit_result = asr_service_module.transcribe_audio(
        "sample.wav",
        language="en",
    )

    service_class.assert_called_once_with()
    assert service.transcribe.call_args_list[0].kwargs == {
        "audio_path": "sample.wav",
        "language": "en",
    }
    assert service.transcribe.call_args_list[1].kwargs == {
        "audio_path": "sample.wav",
        "language": "en",
    }

    for result in (default_result, explicit_result):
        assert result["transcript"] == result["text"]
        assert result["segments"] == service.transcribe.return_value[
            "segments"
        ]
        assert result["language"] == "en"
        assert result["model"] == "base"
        assert result["device"] == "cpu"


def test_asr_service_keeps_vad_aware_transcription_method():
    assert callable(ASRService.transcribe_segments)


def main() -> None:

    print("=" * 70)
    print("GD SHARED ASR TEST")
    print("=" * 70)

    audio_path = (
        "data/gd/audio/p001_test.wav"
    )

    service = ASRService(
        model_name="base"
    )

    result = service.transcribe(
        audio_path=audio_path,
        language="en",
    )

    print()
    print("=" * 70)
    print("ASR RESULT")
    print("=" * 70)

    print()
    print(
        "Detected language:",
        result["language"],
    )

    print(
        "Device:",
        result["device"],
    )

    print(
        "Model:",
        result["model"],
    )

    print()
    print("FULL TRANSCRIPT:")
    print(
        result["text"]
    )

    print()
    print("SEGMENTS:")
    print("-" * 70)

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

        print(
            f"      {segment['text']}"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
