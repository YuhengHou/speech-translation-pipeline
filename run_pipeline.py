from src.main_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        input_audio="data/input/sample.wav",
        output_audio="data/output/sample_zh.wav"
    )