from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base import LM

class HFLocal(LM):
    def __init__(self, hf_name: str, temperature:float, top_p:float, max_new_tokens:int, device="cuda",
                 dtype="float16", gen: dict | None = None, trust_remote_code: bool | None = None):
        self.tok = AutoTokenizer.from_pretrained(
            hf_name, use_fast=True,
            # Qwen/Mistral sometimes need this; harmless for others:
            trust_remote_code=True if trust_remote_code is None else trust_remote_code
        )

        load_kwargs = {"dtype": getattr(torch, dtype)}
        # Only use device_map when we actually want automatic placement (needs accelerate)
        if device in ("auto", "balanced", "sequential"):
            load_kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs)

        # For explicit devices (e.g., "cuda" or "cpu"), move the whole model
        if device not in ("auto", "balanced", "sequential"):
            self.model.to(device)

        self.gen = gen or dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=False)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> str:
        g = dict(self.gen); g["max_new_tokens"] = max_new_tokens
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **g)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
