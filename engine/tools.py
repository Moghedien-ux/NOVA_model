import datetime
import json
import re
from typing import Dict, Any, Tuple
from sympy import SympifyError
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from core.persona import PersistentPersonaMatrix


class NovaToolbox:
    def __init__(self, persona: PersistentPersonaMatrix):
        self.persona = persona
        self.math_transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

    def get_temporal_header(self) -> str:
        now = datetime.datetime.now().astimezone()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")
        return f"<|time|>{formatted}<|end_time|>"

    def evaluate_math(self, expression: str) -> str:
        clean_expr = expression.strip()
        try:
            parsed = parse_expr(clean_expr, transformations=self.math_transformations)
            evaluated = parsed.evalf()
            if evaluated.is_Integer:
                return str(int(evaluated))
            result_str = f"{float(evaluated):.8f}".rstrip("0").rstrip(".")
            return result_str if result_str else "0"
        except (SympifyError, TypeError, ValueError, SyntaxError) as e:
            return f"MathError({e})"

    def process_control_tags(self, text: str, memory_db=None) -> Tuple[str, Dict[str, Any]]:
        events: Dict[str, Any] = {
            "calculations": [],
            "persona_updates": [],
            "learned_words": []
        }

        calc_pattern = re.compile(r"<\|calc\|>(.*?)<\|end_calc\|>", re.DOTALL)

        def _replace_calc(match: re.Match) -> str:
            expr = match.group(1).strip()
            res = self.evaluate_math(expr)
            events["calculations"].append({"expr": expr, "result": res})
            return f"<|calc|>{expr}<|end_calc|><|result|>{res}<|end_result|>"

        resolved_text = calc_pattern.sub(_replace_calc, text)

        mod_pattern = re.compile(r"<\|persona_mod\|>(.*?)<\|end_mod\|>", re.DOTALL)
        for match in mod_pattern.finditer(resolved_text):
            raw_payload = match.group(1).strip()
            try:
                data = json.loads(raw_payload)
                updates = data if isinstance(data, list) else [data]
                for item in updates:
                    trait = str(item.get("trait", "")).strip()
                    if not trait:
                        continue
                    if "delta" in item:
                        new_val = self.persona.modify_trait(trait, float(item["delta"]), relative=True)
                    elif "value" in item:
                        new_val = self.persona.modify_trait(trait, float(item["value"]), relative=False)
                    else:
                        continue
                    events["persona_updates"].append({"trait": trait, "new_value": new_val})
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        word_pattern = re.compile(r"<\|learn_word\|>(.*?)<\|end_word\|>", re.DOTALL)
        for match in word_pattern.finditer(resolved_text):
            raw_word = match.group(1).strip()
            try:
                data = json.loads(raw_word)
                word = str(data.get("word", "")).strip()
                meaning = str(data.get("meaning", "")).strip()
                context_example = str(data.get("context", "")).strip()
                if word and meaning:
                    if memory_db is not None:
                        memory_db.upsert_word(word, meaning, context_example)
                    events["learned_words"].append({"word": word, "meaning": meaning})
            except (json.JSONDecodeError, TypeError):
                pass

        return resolved_text, events

    @staticmethod
    def strip_internal_tags(text: str) -> str:
        text = re.sub(r"<\|persona_mod\|>.*?<\|end_mod\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|learn_word\|>.*?<\|end_word\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|calc\|>.*?<\|end_calc\|>", "", text, flags=re.DOTALL)
        text = text.replace("<|result|>", "").replace("<|end_result|>", "")
        for tok in ["<|bos|>", "<|eos|>", "<|assistant|>", "<|user|>", "<|sys|>"]:
            text = text.replace(tok, "")
        return text.strip()