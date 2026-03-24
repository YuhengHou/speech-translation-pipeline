from pathlib import Path
from TTS.api import TTS


class XTTSCloner:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        print(f"[INFO] Loading XTTS model: {model_name}")
        self.tts = TTS(model_name=model_name)
        print("[INFO] XTTS model loaded.")

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
        language: str = "zh-cn",
    ):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Speaker reference: {speaker_wav}")
        print(f"[INFO] Output language: {language}")
        print(f"[INFO] Output path: {output_file.resolve()}")
        print(f"[INFO] Text: {text}")

        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=str(output_file),
        )

        print("[INFO] XTTS synthesis done.")


if __name__ == "__main__":
    cloner = XTTSCloner()
    cloner.synthesize(
        text="你好，今天天气很好。我正在测试跨语言语音克隆。",
        speaker_wav="data/input/reference.wav",
        output_path="data/output/xtts_test_zh.wav",
        language="zh-cn",
    )