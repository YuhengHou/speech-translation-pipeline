from faster_whisper import WhisperModel


def transcribe_audio(audio_path: str, model_size: str = "medium") -> str:
    """
    Transcribe English audio to text using faster-whisper.
    """
    model = WhisperModel(model_size, compute_type="int8")

    segments, info = model.transcribe(
        audio_path,
        language="en",
        vad_filter=False
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


if __name__ == "__main__":
    result = transcribe_audio("data/input/sample.wav")
    print(result)