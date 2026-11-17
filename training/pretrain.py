import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import NovaConfig
from core.tokenizer import NovaTokenizer
from core.model import NovaModel
from training.dataset_builder import stream_pretrain_texts


class PackedPretrainDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        num_chunks = len(token_ids) // (seq_len + 1)
        trimmed = token_ids[: num_chunks * (seq_len + 1)]
        self.data = torch.tensor(trimmed, dtype=torch.long).view(num_chunks, seq_len + 1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_pretraining(max_docs: int = 30000, epochs: int = 2):
    cfg = NovaConfig()
    cfg.ensure_storage()

    tokenizer = NovaTokenizer(cfg)
    if not os.path.exists(cfg.tokenizer_path):
        print("[Pretrain] Training Nova's Byte-Level BPE tokenizer from FineWeb-Edu stream...")
        tokenizer.train_from_iterator(stream_pretrain_texts(max_samples=15000), length=15000)

    print(f"[Pretrain] Streaming and tokenizing {max_docs} documents...")
    all_tokens = []
    for doc in tqdm(stream_pretrain_texts(max_samples=max_docs), total=max_docs):
        all_tokens.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))

    dataset = PackedPretrainDataset(all_tokens, seq_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    print(f"[Pretrain] Built {len(dataset)} packed sequences of length {cfg.max_seq_len}.")

    model = NovaModel(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Pretrain] Initialized Nova from scratch ({total_params:.2f}M parameters) on {cfg.device}.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay
    )

    neutral_names = ["wit", "compassion", "empathy", "analytical_rigor", "verbosity"]
    neutral_embs = []
    with torch.no_grad():
        for name in neutral_names:
            ids = torch.tensor(tokenizer.encode(name.replace("_", " ")), dtype=torch.long, device=cfg.device)
            neutral_embs.append(model.tok_embeddings(ids).mean(dim=0))
    base_trait_embs = torch.stack(neutral_embs, dim=0).unsqueeze(0)  # (1, 5, d_model)
    base_trait_vals = torch.full((1, 5, 1), 0.5, dtype=torch.float32, device=cfg.device)

    total_steps = (len(loader) * epochs) // cfg.grad_accum_steps
    global_step = 0
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(pbar):
            x, y = x.to(cfg.device), y.to(cfg.device)
            B = x.size(0)
            t_embs = base_trait_embs.expand(B, -1, -1)
            t_vals = base_trait_vals.expand(B, -1, -1)

            with torch.amp.autocast(device_type=cfg.device, dtype=cfg.dtype, enabled=(cfg.device == "cuda")):
                _, loss, _ = model(x, t_embs, t_vals, targets=y)
                scaled_loss = loss / cfg.grad_accum_steps

            scaled_loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                lr = get_cosine_lr(global_step, cfg.warmup_steps, total_steps, cfg.pretrain_lr, cfg.pretrain_lr * 0.1)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        model.save_base_weights(cfg.base_weights_path)
        print(f"[Pretrain] Saved foundational weights to {cfg.base_weights_path}")


if __name__ == "__main__":
    run_pretraining()