import torch
from config import NovaConfig
from core.model import NovaModel
from core.tokenizer import NovaTokenizer
from core.persona import PersistentPersonaMatrix
from engine.memory import PersistentMemoryDB


class LiveSynapticLearner:
    def __init__(
        self,
        cfg: NovaConfig,
        model: NovaModel,
        tokenizer: NovaTokenizer,
        persona: PersistentPersonaMatrix,
        memory_db: PersistentMemoryDB
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.persona = persona
        self.memory_db = memory_db
        self.model.enable_live_plasticity()
        plastic_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            plastic_params,
            lr=cfg.live_learning_rate,
            weight_decay=0.01
        )

    def _encode_training_pair(self, formatted_sequence: str):
        ids = self.tokenizer.encode(formatted_sequence, add_bos=True, add_eos=True)
        if len(ids) < 4:
            return None, None
        ids = ids[:self.cfg.max_seq_len]
        tensor_ids = torch.tensor(ids, dtype=torch.long, device=self.cfg.device).unsqueeze(0)
        x = tensor_ids[:, :-1]
        y = tensor_ids[:, 1:]
        return x, y

    def adapt_on_turn(self, formatted_turn: str) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        trait_embs, trait_vals = self.persona.to_tensors(
            self.tokenizer, self.model.tok_embeddings, self.cfg.device
        )
        x_live, y_live = self._encode_training_pair(formatted_turn)
        if x_live is None:
            return 0.0

        with torch.amp.autocast(device_type=self.cfg.device, dtype=self.cfg.dtype, enabled=(self.cfg.device == "cuda")):
            _, loss_live, _ = self.model(x_live, trait_embs, trait_vals, targets=y_live)
            replay_samples = self.memory_db.sample_replay(count=1)
            if replay_samples:
                x_rep, y_rep = self._encode_training_pair(replay_samples[0])
                if x_rep is not None:
                    _, loss_rep, _ = self.model(x_rep, trait_embs, trait_vals, targets=y_rep)
                    total_loss = (1.0 - self.cfg.replay_mix_ratio) * loss_live + self.cfg.replay_mix_ratio * loss_rep
                else:
                    total_loss = loss_live
            else:
                total_loss = loss_live

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        self.optimizer.step()
        self.model.save_plastic_weights(self.cfg.plastic_weights_path)
        self.model.eval()
        return float(total_loss.item())