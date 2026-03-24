from src.asr import transcribe_audio
from src.mt import NLLBTranslator
from src.llm_refine import refine_asr_text
from src.xtts_clone import XTTSCloner


def run_pipeline(input_audio: str, reference_audio: str):
    print("[INFO] Step 1: Starting ASR...")
    transcript = transcribe_audio(input_audio)
    print("\n[ASR Transcript]")
    print(transcript)

    print("\n[INFO] Step 2: Starting LLM refinement...")
    refined_text = refine_asr_text(transcript)
    print("\n[LLM Refined Transcript]")
    print(refined_text)

    print("\n[INFO] Step 3: Loading MT model...")
    translator = NLLBTranslator(src_lang="eng_Latn", tgt_lang="zho_Hans")

    print("\n[INFO] Step 4: Translating...")
    translated_text = translator.translate(refined_text)
    print("\n[Translated Text]")
    print(translated_text)

    print("\n[INFO] Step 5: Loading XTTS...")
    xtts = XTTSCloner()

    print("\n[INFO] Step 6: Generating cloned Chinese speech...")
    xtts.synthesize(
        text=translated_text,
        speaker_wav=reference_audio,
        output_path="data/output/final_xtts_zh.wav",
        language="zh-cn",
    )

    return transcript, refined_text, translated_text


if __name__ == "__main__":
    run_pipeline(
        input_audio="data/input/sample.wav",
        reference_audio="data/input/reference.wav",
    )