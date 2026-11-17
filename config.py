import os
from dataclasses import dataclass, field
import torch

@dataclass
class NovaConfig:
    # --- Project & Storage Paths ---
    storage_dir: str = "storage"
    tokenizer_path: str = "storage/nova_tokenizer.json"
    base_weights_path: str = "storage/nova_base.pt"
    plastic_weights_path: str = "storage/nova_plastic.pt"
    persona_path: str = "storage/nova_persona.json"
    memory_db_path: str = "storage/nova_mind.db"

    # --- Transformer Architecture ---
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.05
    rope_theta: float = 10000.0

    # --- 1-to-1 Synaptic Plasticity (LoRA) ---
    lora_rank: int = 16
    lora_alpha: float = 32.0
    live_learning_rate: float = 1e-4
    replay_mix_ratio: float = 0.50  # 50% core replay data mixed with live conversation to prevent forgetting

    # --- Offline Pretraining & SFT ---
    batch_size: int = 8
    grad_accum_steps: int = 4
    pretrain_lr: float = 3e-4
    sft_lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000

    # --- Special Control Tokens ---
    special_tokens: list[str] = field(default_factory=lambda: [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|sys|>",
        "<|user|>",
        "<|assistant|>",
        "<|time|>",
        "<|end_time|>",
        "<|calc|>",
        "<|end_calc|>",
        "<|result|>",
        "<|end_result|>",
        "<|persona_mod|>",
        "<|end_mod|>",
        "<|learn_word|>",
        "<|end_word|>"
    ])

    # --- Hardware Selection ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )

    def ensure_storage(self):
        os.makedirs(self.storage_dir, exist_ok=True)