import os

from faster_whisper import WhisperModel


MODEL_NAME = "small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_model = None


def get_asr_model():
    """
    Load the Whisper model only when it is first needed.
    """

    global _model

    if _model is None:
        print("Loading Whisper ASR model...")

        _model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

        print("Whisper ASR model loaded.")

    return _model


def transcribe_audio(audio_path, language="en"):
    """
    Transcribe an audio file using faster-whisper.
    """

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model = get_asr_model()

    segments, info = model.transcribe(
        audio_path,
        language=language
    )

    segment_list = []
    full_transcript = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            }

            segment_list.append(segment_data)
            full_transcript.append(text)

    transcript = " ".join(full_transcript)

    return {
        "transcript": transcript,
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4
        ),
        "duration": round(info.duration, 2),
        "segments": segment_list
    }