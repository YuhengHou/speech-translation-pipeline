# Robust and Speaker-Preserving Speech Translation with Generative Models

## Overview

This project builds an end-to-end speech translation system that improves both

- Content accuracy using Large Language Models (LLMs)
- Speaker identity preservation using cross-lingual voice cloning

The goal is to move beyond traditional pipelines by leveraging generative models to produce more accurate and realistic translated speech.

---

## Pipeline

The system processes input speech as follows


Audio (English)
→ ASR (Whisper)
→ LLM (Qwen2.5)
→ Translation (MT or LLM)
→ XTTS v2 (Voice Cloning)
→ Audio (Chinese, same speaker)


---

## Implemented Pipelines

### Pipeline A (Baseline)

ASR → MT → XTTS


### Pipeline B (LLM Refinement)

ASR → LLM (refine) → MT → XTTS


### Pipeline C (LLM Translation)

ASR → LLM (translate) → XTTS


---

## Key Components

### 1. ASR (Speech-to-Text)
- Model `faster-whisper`
- Converts speech into raw transcript

---

### 2. LLM Module (Qwen2.5 7B)
Used as an intermediate generative layer

#### a) Transcript Refinement
- Fix ASR errors
- Restore punctuation
- Normalize text
- Preserve meaning

Example

Input this is bailey the meeting is at to or three
Output This is Billy. The meeting is at two or three.


#### b) Direct Translation
- Translates transcript directly to Chinese
- More robust to noisy input

---

### 3. Chunk-Based Processing
- Audio split into overlapping segments
- Reduces boundary truncation errors
- Enables long audio processing

---

### 4. Memory Mechanism (Planned  Partial)
Maintains cross-chunk consistency

```python
memory = {
  entities [Billy, Qwen2.5],
  glossary {speech translation 语音翻译},
  context The speaker is discussing a project.
}

Improves

name consistency
terminology stability
discourse coherence
5. Translation (Baseline)
Model NLLB-200 (600M)
Used for controlled comparison with LLM translation
6. XTTS v2 (Voice Cloning)
Cross-lingual speech generation
Input
translated text
reference speaker audio
Output
translated speech with preserved voice characteristics
Project Structure
speech-translation-pipeline
│
├── src
│   ├── asr.py
│   ├── llm_refine.py
│   ├── mt.py
│   ├── xtts_clone.py
│   ├── main_pipeline.py
│
├── data
│   ├── input
│   ├── output
│
├── run_pipeline.py
├── README.md
Setup
1. Create environment
python -m venv .venv
.venvScriptsactivate
2. Install dependencies
pip install torch torchaudio transformers sentencepiece soundfile librosa faster-whisper
pip install coqui-tts pypinyin
3. Install Ollama (for LLM)

Run

ollama run qwen2.5latest
Usage
Run full pipeline
python run_pipeline.py
XTTS standalone test
python srcxtts_clone.py
Evaluation

We compare

ASR → MT
ASR → LLM → MT
ASR → LLM → XTTS

Metrics

WER (transcription accuracy)
BLEU (translation quality)
qualitative analysis
name consistency
error correction
speaker similarity
Motivation

Traditional speech translation

propagates ASR errors
loses speaker identity

This project explores

Using generative models to improve both meaning and voice in speech translation.

Future Work
Full memory-aware LLM refinement
Speaker similarity metrics
Prosody alignment analysis
Real-time streaming pipeline
XTTS tuning  prompt control
Author

Yuheng Hou
Johns Hopkins University
ECE (Human Language Technology)

License

MIT License

