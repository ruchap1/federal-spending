"""Local Hugging Face answer generation."""

from __future__ import annotations

import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL = "google/flan-t5-base"


class LocalGenerator:
    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or os.environ.get("RAG_GENERATOR_MODEL", DEFAULT_MODEL)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation="eager")
        self.model.eval()
        self.last_prompt = ""
        self.last_answer = ""
        self.max_input_tokens = 512

    def answer_documents(self, question, results):
        """Allocate the input budget before generation; preserve the question."""
        from src.prompts import build_budgeted_prompt
        prompt = build_budgeted_prompt(question, results, self.tokenizer, self.max_input_tokens)
        return self.generate(prompt)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        self.last_prompt = prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[-1] > self.max_input_tokens:
            raise ValueError("Prompt exceeds the input budget; use answer_documents().")
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=2,
                no_repeat_ngram_size=4,
            )
        self.last_answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return self.last_answer

