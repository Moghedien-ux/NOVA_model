import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import NovaConfig
from core.tokenizer import NovaTokenizer
from core.model import NovaModel
from engine.memory import PersistentMemoryDB
from training.dataset_builder import build_gsm8k_tool_samples, build_cognitive_sft_samples


class NovaSFTDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer: NovaTokenizer, max_seq_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ids = self.tokenizer.encode(item["text"], add_bos=True, add_eos=True)
        ids = ids[: self.max_seq_len]

        pad_len = self.max_seq_len - len(ids)
        input_ids = ids + [self.tokenizer.pad_id] * pad_len

        tensor_ids = torch.tensor(input_ids, dtype=torch.long)
        x = tensor_ids[:-1]
        y = tensor_ids[1:].clone()
        y[y == self.tokenizer.pad_id] = -100

        return x, y, item["traits"]


def sft_collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    traits_list = [b[2] for b in batch]
    return xs, ys, traits_list


def encode_batch_traits(
    traits_list: list[dict],
    tokenizer: NovaTokenizer,
    embed_layer: torch.nn.Embedding,
    device: str
):
    max_traits = max(len(t) for t in traits_list)
    d_model = embed_layer.embedding_dim
    B = len(traits_list)

    batch_embs = torch.zeros((B, max_traits, d_model), dtype=torch.float32, device=device)
    batch_vals = torch.full((B, max_traits, 1), 0.50, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, traits in enumerate(traits_list):
            for j, (name, val) in enumerate(traits.items()):
                ids = torch.tensor(
                    tokenizer.encode(name.replace("_", " ")),
                    dtype=torch.long,
                    device=device
                )
                batch_embs[i, j] = embed_layer(ids).mean(dim=0)
                batch_vals[i, j, 0] = float(val)

    return batch_embs, batch_vals


def run_sft(epochs: int = 4):
    cfg = NovaConfig()
    cfg.ensure_storage()

    tokenizer = NovaTokenizer(cfg)
    model = NovaModel(cfg).to(cfg.device)

    if os.path.exists(cfg.base_weights_path):
        print(f"[SFT] Loading pre-trained base weights from {cfg.base_weights_path}...")
        model.load_base_weights(cfg.base_weights_path, cfg.device)
    else:
        print("[SFT] Warning: No base weights found. Training SFT from current initialization.")

    math_samples = build_gsm8k_tool_samples(max_samples=5000)
    cognitive_samples = build_cognitive_sft_samples()
    all_samples = math_samples + cognitive_samples
    random.shuffle(all_samples)

    dataset = NovaSFTDataset(all_samples, tokenizer, max_seq_len=512)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.sft_lr,
        weight_decay=0.05
    )

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"SFT Epoch {epoch + 1}/{epochs}")
        for x, y, traits_list in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)
            t_embs, t_vals = encode_batch_traits(traits_list, tokenizer, model.tok_embeddings, cfg.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=cfg.device, dtype=cfg.dtype, enabled=(cfg.device == "cuda")):
                _, loss, _ = model(x, t_embs, t_vals, targets=y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.save_base_weights(cfg.base_weights_path)
    print(f"[SFT] Updated foundational weights saved to {cfg.base_weights_path}")

    memory_db = PersistentMemoryDB(cfg.memory_db_path)
    replay_anchors = [s["text"] for s in random.sample(all_samples, min(500, len(all_samples)))]
    memory_db.seed_replay_buffer(replay_anchors)
    memory_db.close()
    print(f"[SFT] Seeded {len(replay_anchors)} replay anchors into {cfg.memory_db_path}")


if __name__ == "__main__":
    run_sft()