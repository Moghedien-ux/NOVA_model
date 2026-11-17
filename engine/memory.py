import datetime
import os
import sqlite3
from typing import List, Tuple, Optional
import numpy as np
import torch


class PersistentMemoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=FULL;")
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    nova_text TEXT NOT NULL,
                    persona_snapshot TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_lexicon (
                    word TEXT PRIMARY KEY,
                    meaning TEXT NOT NULL,
                    context_example TEXT,
                    encounters INTEGER DEFAULT 1,
                    last_updated TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS replay_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    formatted_sample TEXT NOT NULL
                )
            """)

    def store_turn(
        self,
        user_text: str,
        nova_text: str,
        persona_summary: str,
        embedding: torch.Tensor
    ):
        now = datetime.datetime.now().astimezone().isoformat()
        vec_np = embedding.detach().cpu().float().numpy().flatten().astype(np.float32)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO episodic_memories (timestamp, user_text, nova_text, persona_snapshot, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, user_text, nova_text, persona_summary, vec_np.tobytes())
            )

    def retrieve_relevant_memories(self, query_embedding: torch.Tensor, top_k: int = 3) -> List[Tuple[str, str, str]]:
        cursor = self.conn.execute(
            "SELECT timestamp, user_text, nova_text, embedding FROM episodic_memories"
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        q_vec = query_embedding.detach().cpu().float().numpy().flatten().astype(np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-8

        scored = []
        for ts, u_txt, n_txt, emb_blob in rows:
            m_vec = np.frombuffer(emb_blob, dtype=np.float32)
            if m_vec.shape != q_vec.shape:
                continue
            sim = float(np.dot(q_vec, m_vec) / (q_norm * (np.linalg.norm(m_vec) + 1e-8)))
            scored.append((sim, ts, u_txt, n_txt))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(ts, u, n) for _, ts, u, n in scored[:top_k]]

    def get_recent_turns(self, limit: int = 4) -> List[Tuple[str, str]]:
        cursor = self.conn.execute(
            "SELECT user_text, nova_text FROM episodic_memories ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        return list(reversed(rows))

    def upsert_word(self, word: str, meaning: str, context_example: str = ""):
        clean_word = word.lower().strip()
        now = datetime.datetime.now().astimezone().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learned_lexicon (word, meaning, context_example, encounters, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(word) DO UPDATE SET
                    meaning = excluded.meaning,
                    context_example = CASE
                        WHEN excluded.context_example != '' THEN excluded.context_example
                        ELSE learned_lexicon.context_example
                    END,
                    encounters = learned_lexicon.encounters + 1,
                    last_updated = excluded.last_updated
                """,
                (clean_word, meaning, context_example, now)
            )

    def lookup_words_in_text(self, text: str) -> List[Tuple[str, str]]:
        tokens = set(re_split := [w.strip(".,!?;:\"'()[]{}").lower() for w in text.split()])
        if not tokens:
            return []
        placeholders = ",".join("?" for _ in tokens)
        cursor = self.conn.execute(
            f"SELECT word, meaning FROM learned_lexicon WHERE word IN ({placeholders})",
            tuple(tokens)
        )
        return cursor.fetchall()

    def seed_replay_buffer(self, samples: List[str]):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO replay_buffer (formatted_sample) VALUES (?)",
                [(s,) for s in samples]
            )

    def sample_replay(self, count: int = 1) -> List[str]:
        cursor = self.conn.execute(
            "SELECT formatted_sample FROM replay_buffer ORDER BY RANDOM() LIMIT ?",
            (count,)
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        self.conn.close()