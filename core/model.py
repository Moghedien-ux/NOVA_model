import os
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NovaConfig
from core.layers import NovaBlock, RMSNorm, precompute_rope_freqs
from core.persona import PersonaMatrixEncoder


class NovaModel(nn.Module):
    def __init__(self, cfg: NovaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.persona_encoder = PersonaMatrixEncoder(cfg.d_model)
        self.layers = nn.ModuleList([NovaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying between token embeddings and LM output head
        self.lm_head.weight = self.tok_embeddings.weight

        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta),
            persistent=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            if module.weight.abs().sum() != 0:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        trait_embeddings: torch.Tensor,
        trait_values: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len:]
            if targets is not None:
                targets = targets[:, -self.cfg.max_seq_len:]
            T = self.cfg.max_seq_len

        x = self.tok_embeddings(idx)
        z_p = self.persona_encoder(trait_embeddings, trait_values)
        freqs_cis = self.freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis, z_p)

        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss, hidden_states

    @torch.no_grad()
    def extract_embedding(
        self,
        idx: torch.Tensor,
        trait_embeddings: torch.Tensor,
        trait_values: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        _, _, hidden_states = self.forward(idx, trait_embeddings, trait_values)
        pooled = hidden_states.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        trait_embeddings: torch.Tensor,
        trait_values: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_token_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        self.eval()
        stop_set = set(stop_token_ids or [])

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _, _ = self.forward(idx_cond, trait_embeddings, trait_values)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() in stop_set:
                break

        return idx

    def enable_live_plasticity(self):
        for name, param in self.named_parameters():
            param.requires_grad = ("lora_" in name)

    def save_base_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_base_weights(self, path: str, device: str):
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state, strict=False)

    def save_plastic_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plastic = {k: v.cpu() for k, v in self.state_dict().items() if "lora_" in k}
        torch.save(plastic, path)

    def load_plastic_weights(self, path: str, device: str):
        if os.path.exists(path):
            plastic = torch.load(path, map_location=device, weights_only=True)
            self.load_state_dict(plastic, strict=False)