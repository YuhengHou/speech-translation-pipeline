from src.asr import transcribe_audio
from src.mt import NLLBTranslator


def run_pipeline(input_audio: str):
    # 1. ASR
    transcript = transcribe_audio(input_audio)
    print("\n[ASR Transcript]")
    print(transcript)

    # 2. MT
    translator = NLLBTranslator(src_lang="eng_Latn", tgt_lang="zho_Hans")
    translated_text = translator.translate(transcript)

    print("\n[Translated Text]")
    print(translated_text)

    return transcript, translated_text


if __name__ == "__main__":
    run_pipeline("data/input/sample.wav")