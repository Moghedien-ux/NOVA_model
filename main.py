import os
import re
import sys
import torch

from config import NovaConfig
from core.tokenizer import NovaTokenizer
from core.model import NovaModel
from core.persona import PersistentPersonaMatrix
from engine.tools import NovaToolbox
from engine.memory import PersistentMemoryDB
from engine.learner import LiveSynapticLearner


class NovaRuntime:
    def __init__(self):
        self.cfg = NovaConfig()
        self.cfg.ensure_storage()

        if not os.path.exists(self.cfg.tokenizer_path) or not os.path.exists(self.cfg.base_weights_path):
            print("[Nova] Foundational brain not found on disk.")
            print("       Run `python -m training.pretrain` followed by `python -m training.sft` first.")
            sys.exit(1)

        self.tokenizer = NovaTokenizer(self.cfg)
        self.persona = PersistentPersonaMatrix(self.cfg.persona_path)
        self.memory_db = PersistentMemoryDB(self.cfg.memory_db_path)
        self.toolbox = NovaToolbox(self.persona)

        self.model = NovaModel(self.cfg).to(self.cfg.device)
        self.model.load_base_weights(self.cfg.base_weights_path, self.cfg.device)
        self.model.load_plastic_weights(self.cfg.plastic_weights_path, self.cfg.device)

        self.learner = LiveSynapticLearner(
            self.cfg, self.model, self.tokenizer, self.persona, self.memory_db
        )

        self.eos_id = self.tokenizer.eos_id
        self.end_calc_id = self.tokenizer.token_to_id("<|end_calc|>")

    def _build_context_prompt(self, user_input: str, query_emb: torch.Tensor) -> str:
        temporal_header = self.toolbox.get_temporal_header()

        known_words = self.memory_db.lookup_words_in_text(user_input)
        lexicon_block = ""
        if known_words:
            defs = "; ".join(f"{w}: {m}" for w, m in known_words)
            lexicon_block = f"\n[Learned Lexicon: {defs}]"

        relevant = self.memory_db.retrieve_relevant_memories(query_emb, top_k=2)
        memory_block = ""
        if relevant:
            mem_lines = [f"- ({ts[:16]}) User: {u} | Nova: {n}" for ts, u, n in relevant]
            memory_block = "\n[Episodic Recall:\n" + "\n".join(mem_lines) + "]"

        recent_turns = self.memory_db.get_recent_turns(limit=3)
        history_block = ""
        for u_prev, n_prev in recent_turns:
            history_block += f"<|user|>{u_prev}<|eos|>\n<|assistant|>{n_prev}<|eos|>\n"

        prompt = (
            f"<|sys|>{temporal_header}\n"
            f"You are Nova, an adaptive AI assistant. Active traits: [{self.persona.summary()}]."
            f"{lexicon_block}{memory_block}<|eos|>\n"
            f"{history_block}"
            f"<|user|>{user_input}<|eos|>\n"
            f"<|assistant|>"
        )
        return prompt

    def _generate_with_live_tools(self, prompt: str, max_steps: int = 4) -> str:
        current_text = prompt
        generated_assistant_part = ""

        for _ in range(max_steps):
            trait_embs, trait_vals = self.persona.to_tensors(
                self.tokenizer, self.model.tok_embeddings, self.cfg.device
            )
            input_ids = self.tokenizer.encode(current_text, add_bos=True, add_eos=False)
            idx = torch.tensor([input_ids], dtype=torch.long, device=self.cfg.device)

            out_ids = self.model.generate(
                idx=idx,
                trait_embeddings=trait_embs,
                trait_values=trait_vals,
                max_new_tokens=200,
                temperature=0.65,
                top_p=0.90,
                stop_token_ids=[self.eos_id, self.end_calc_id]
            )

            new_tokens = out_ids[0, len(input_ids):].tolist()
            chunk_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            generated_assistant_part += chunk_text
            current_text += chunk_text
            if "<|calc|>" in chunk_text and "<|end_calc|>" in chunk_text:
                calc_match = re.search(r"<\|calc\|>(.*?)<\|end_calc\|>$", generated_assistant_part, re.DOTALL)
                if calc_match:
                    expr = calc_match.group(1).strip()
                    result = self.toolbox.evaluate_math(expr)
                    result_injection = f"<|result|>{result}<|end_result|>"
                    generated_assistant_part += result_injection
                    current_text += result_injection
                    continue
            break

        return generated_assistant_part

    def chat_turn(self, user_input: str) -> str:
        trait_embs, trait_vals = self.persona.to_tensors(
            self.tokenizer, self.model.tok_embeddings, self.cfg.device
        )
        u_ids = torch.tensor(
            [self.tokenizer.encode(user_input, add_bos=True)],
            dtype=torch.long,
            device=self.cfg.device
        )
        query_emb = self.model.extract_embedding(u_ids, trait_embs, trait_vals)

        prompt = self._build_context_prompt(user_input, query_emb)

        raw_response = self._generate_with_live_tools(prompt)

        resolved_response, events = self.toolbox.process_control_tags(
            raw_response, memory_db=self.memory_db
        )
        clean_reply = self.toolbox.strip_internal_tags(resolved_response)

        self.memory_db.store_turn(
            user_text=user_input,
            nova_text=clean_reply,
            persona_summary=self.persona.summary(),
            embedding=query_emb
        )

        training_trace = (
            f"<|sys|>{self.toolbox.get_temporal_header()}\n"
            f"You are Nova, an adaptive AI assistant.<|eos|>\n"
            f"<|user|>{user_input}<|eos|>\n"
            f"<|assistant|>{resolved_response}<|eos|>"
        )
        plastic_loss = self.learner.adapt_on_turn(training_trace)

        if events["persona_updates"]:
            print(f"  [Persona Matrix Updated] -> {self.persona.summary()}")
        if events["learned_words"]:
            for w in events["learned_words"]:
                print(f"  [Lexicon Acquired] -> '{w['word']}': {w['meaning']}")
        print(f"  [1-to-1 Synaptic Commit | Loss: {plastic_loss:.4f}]")

        return clean_reply

    def run_cli(self):
        print("=" * 64)
        print(" NOVA ONLINE | 1-to-1 Persistent Cognitive Architecture")
        print(f" Clock:  {self.toolbox.get_temporal_header()}")
        print(f" Traits: {self.persona.summary()}")
        print(" Commands: /traits (inspect matrix) | /quit (exit)")
        print("=" * 64)

        try:
            while True:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"/quit", "/exit"}:
                    break
                if user_input.lower() == "/traits":
                    print(f"[Persona Matrix] {self.persona.summary()}")
                    continue

                reply = self.chat_turn(user_input)
                print(f"Nova: {reply}")
        finally:
            self.memory_db.close()
            print("\n[Nova] Session closed. All weights, traits, and memories persisted 1-to-1.")


if __name__ == "__main__":
    nova = NovaRuntime()
    nova.run_cli()