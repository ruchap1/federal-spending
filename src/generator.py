"""Local Hugging Face answer generation."""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL = "google/flan-t5-small"


class LocalGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=2,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

