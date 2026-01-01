"""
LLM wrappers for BELLE framework.

This project is designed to run fully offline with a local causal language model
(e.g., Qwen3). The wrapper is intentionally lightweight and exposes a single
`generate(prompt, **kwargs)` interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

import os
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text given a prompt."""
        raise NotImplementedError


@dataclass
class TransformersCausalLMConfig:
    model_path: str
    temperature: float = 0.7
    max_tokens: int = 512
    trust_remote_code: bool = True
    device_map: str = "auto"  # "auto" / "cpu" / "cuda"
    torch_dtype: str = "auto"  # "auto" / "float16" / "bfloat16" / "float32"
    repetition_penalty: float = 1.05
    use_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant. Follow format constraints strictly. If asked for JSON, output JSON only."

class TransformersCausalLM(BaseLLM):
    """
    A simple wrapper around a local HF causal LM.

    Notes:
    - Uses `model.generate` directly for predictability.
    - Works offline if the model_path is a local directory.
    """

    def __init__(self, cfg: TransformersCausalLMConfig):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is not available. Please install transformers.")

        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"Model path does not exist: {cfg.model_path}")

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=cfg.trust_remote_code)

        # Some models do not have pad_token set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = None
        if cfg.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, cfg.torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            device_map=cfg.device_map if cfg.device_map != "cpu" else None
        )

        if cfg.device_map == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = int(kwargs.get("max_tokens", self.cfg.max_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))
        repetition_penalty = float(kwargs.get("repetition_penalty", self.cfg.repetition_penalty))

        # ---- build input text (chat template for instruct models) ----
        use_chat = bool(getattr(self.cfg, "use_chat_template", True)) and hasattr(self.tokenizer, "apply_chat_template")
        if use_chat:
            messages = [
                {"role": "system", "content": getattr(self.cfg, "system_prompt", "")},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature > 0.0
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(1e-5, temperature) if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # ---- decode ONLY newly generated tokens (avoid prompt-echo) ----
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def create_llm(config: Dict[str, Any]) -> BaseLLM:
    """
    Factory. Currently supports a local HF causal LM.
    Expected config format (config.yaml):

    llm:
      model_path: "/path/to/model"
      temperature: 0.7
      max_tokens: 512
      trust_remote_code: true
      device_map: "auto"
      torch_dtype: "auto"
    """
    llm_cfg = config.get("llm", {})
    model_path = llm_cfg.get("model_path")
    if not model_path:
        raise ValueError("Missing llm.model_path in config.")

    cfg = TransformersCausalLMConfig(
        model_path=model_path,
        temperature=float(llm_cfg.get("temperature", 0.7)),
        max_tokens=int(llm_cfg.get("max_tokens", 512)),
        trust_remote_code=bool(llm_cfg.get("trust_remote_code", True)),
        device_map=str(llm_cfg.get("device_map", "auto")),
        torch_dtype=str(llm_cfg.get("torch_dtype", "auto")),
        repetition_penalty=float(llm_cfg.get("repetition_penalty", 1.05)),
    )
    return TransformersCausalLM(cfg)
