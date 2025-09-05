from typing import Protocol

class LM(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 8) -> str: ...
