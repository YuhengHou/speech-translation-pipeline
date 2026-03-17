from pathlib import Path
from TTS.api import TTS


class ChineseTTS:
    def __init__(self, model_name: str):
        print(f"[INFO] Loading TTS model: {model_name}")
        self.tts = TTS(model_name=model_name)
        print("[INFO] TTS model loaded.")

    def synthesize(self, text: str, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Synthesizing text: {text}")
        print(f"[INFO] Saving to: {output_file.resolve()}")

        self.tts.tts_to_file(
            text=text,
            file_path=str(output_file)
        )

        print("[INFO] Synthesis done.")


if __name__ == "__main__":
    model_name = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
    output_path = "data/output/test_zh.wav"

    tts = ChineseTTS(model_name)
    tts.synthesize("你好，今天天气真的很糟糕。", output_path)

    print(f"[INFO] Expected output file: {Path(output_path).resolve()}")