import os
from typing import Iterator, List, Optional
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from config import NovaConfig

class NovaTokenizer:
    def __init__(self, cfg: NovaConfig):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None

        if os.path.exists(cfg.tokenizer_path):
            self.load(cfg.tokenizer_path)

    def _init_empty_tokenizer(self) -> Tokenizer:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
        return tok

    def train_from_files(self, files: List[str]):
        self.cfg.ensure_storage()
        tok = self._init_empty_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=self.cfg.vocab_size,
            min_frequency=2,
            special_tokens=self.cfg.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tok.train(files, trainer=trainer)
        tok.save(self.cfg.tokenizer_path)
        self.tokenizer = tok
        print(f"[NovaTokenizer] Trained and saved to {self.cfg.tokenizer_path} (Vocab: {self.vocab_size})")

    def train_from_iterator(self, text_iterator: Iterator[str], length: Optional[int] = None):
        self.cfg.ensure_storage()
        tok = self._init_empty_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=self.cfg.vocab_size,
            min_frequency=2,
            special_tokens=self.cfg.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tok.train_from_iterator(text_iterator, trainer=trainer, length=length)
        tok.save(self.cfg.tokenizer_path)
        self.tokenizer = tok
        print(f"[NovaTokenizer] Trained from stream and saved (Vocab: {self.vocab_size})")

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded yet.")
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded yet.")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded yet.")
        tid = self.tokenizer.token_to_id(token)
        if tid is None:
            raise ValueError(f"Token '{token}' not found in Nova's vocabulary.")
        return tid

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else self.cfg.vocab_size

    @property
    def pad_id(self) -> int:
        return self.token_to_id("<|pad|>")

    @property
    def bos_id(self) -> int:
        return self.token_to_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self.token_to_id("<|eos|>")