import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"

PROMPT_TEMPLATE = """You are an ASR transcript refinement assistant.

Task:
Refine the following ASR transcript.

Rules:
1. Correct obvious ASR recognition errors.
2. Restore punctuation and capitalization.
3. Normalize text where appropriate.
4. Preserve the original meaning.
5. Do NOT translate.
6. Do NOT add information.
7. Return only the refined transcript.

ASR transcript:
{asr_text}
"""

def refine_asr_text(asr_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(asr_text=asr_text)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()