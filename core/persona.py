import json
import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
from core.tokenizer import NovaTokenizer


class PersonaMatrixEncoder(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, trait_embeddings: torch.Tensor, trait_values: torch.Tensor) -> torch.Tensor:
        centered = (trait_values - 0.5) * 2.0
        pooled = (trait_embeddings * centered).mean(dim=1, keepdim=True)
        return self.net(pooled)


class PersistentPersonaMatrix:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        if os.path.exists(storage_path):
            with open(storage_path, "r", encoding="utf-8") as f:
                self.traits: Dict[str, float] = json.load(f)
        else:
            # Extensible initial traits; new traits can be added dynamically at runtime
            self.traits = {
                "wit": 0.75,
                "compassion": 0.65,
                "empathy": 0.65,
                "analytical_rigor": 0.80,
                "verbosity": 0.50
            }
            self.save()

    def modify_trait(self, trait: str, value: float, relative: bool = True) -> float:
        key = trait.lower().strip().replace(" ", "_")
        current = self.traits.get(key, 0.50)
        updated = (current + value) if relative else value
        updated = round(max(0.0, min(1.0, updated)), 4)
        self.traits[key] = updated
        self.save()
        return updated

    def save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.traits, f, indent=2)

    def to_tensors(
        self,
        tokenizer: NovaTokenizer,
        embed_layer: nn.Embedding,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        names = list(self.traits.keys())
        values = torch.tensor(
            [[self.traits[k]] for k in names],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)  # (1, K, 1)

        embs = []
        with torch.no_grad():
            for name in names:
                clean_name = name.replace("_", " ")
                ids = tokenizer.encode(clean_name)
                id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
                # Mean-pool subword embeddings so any new trait word generalizes seamlessly
                embs.append(embed_layer(id_tensor).mean(dim=0))

        trait_embeddings = torch.stack(embs, dim=0).unsqueeze(0)  # (1, K, d_model)
        return trait_embeddings, values

    def summary(self) -> str:
        return ", ".join(f"{k}: {int(v * 100)}%" for k, v in self.traits.items())