from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"


class NLLBTranslator:
    def __init__(self, src_lang: str = "eng_Latn", tgt_lang: str = "zho_Hans"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            src_lang=src_lang
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def translate(self, text: str, max_length: int = 256) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
            max_length=max_length
        )

        return self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]


if __name__ == "__main__":
    translator = NLLBTranslator()
    text = "Hello, this is a test of the speech translation pipeline."
    zh_text = translator.translate(text)
    print(zh_text)