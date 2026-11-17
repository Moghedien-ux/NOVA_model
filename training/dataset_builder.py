import datetime
import json
import random
import re
from typing import List, Dict, Tuple, Iterator
from datasets import load_dataset


def stream_pretrain_texts(max_samples: int = 50000) -> Iterator[str]:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    count = 0
    for row in dataset:
        text = row.get("text", "").strip()
        if len(text) > 200:
            yield text
            count += 1
            if count >= max_samples:
                break


def build_gsm8k_tool_samples(max_samples: int = 7000) -> List[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    samples = []

    calc_regex = re.compile(r"<<([^=<>]+)=([^<>]+)>>")

    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        question = row["question"].strip()
        raw_answer = row["answer"].strip()

        def _convert_tag(m: re.Match) -> str:
            expr = m.group(1).strip()
            res = m.group(2).strip()
            return f"<|calc|>{expr}<|end_calc|><|result|>{res}<|end_result|>"

        formatted_steps = calc_regex.sub(_convert_tag, raw_answer)
        formatted_steps = formatted_steps.replace("####", "\nFinal Answer:")
        traits = {
            "wit": round(random.uniform(0.2, 0.8), 2),
            "compassion": 0.50,
            "empathy": 0.50,
            "analytical_rigor": round(random.uniform(0.75, 0.98), 2),
            "verbosity": round(random.uniform(0.3, 0.7), 2)
        }

        text = (
            f"<|sys|><|time|>2026-11-15 14:30:00 UTC (Sunday)<|end_time|>\n"
            f"You are Nova, an adaptive AI assistant.<|eos|>\n"
            f"<|user|>{question}<|eos|>\n"
            f"<|assistant|>{formatted_steps}<|eos|>"
        )
        samples.append({"text": text, "traits": traits})

    return samples


def build_cognitive_sft_samples() -> List[Dict]:
    samples = []

    contrastive_scenarios = [
        {
            "user": "A runaway railcar is heading toward five track workers. You can pull a lever to divert it onto a side track where one worker is standing. What should be done?",
            "variants": [
                (
                    {"wit": 0.15, "compassion": 0.95, "empathy": 0.95, "analytical_rigor": 0.40},
                    "That is a deeply painful moral tragedy where every choice carries devastating human grief. "
                    "While diverting the car saves five lives, intentionally redirecting harm onto an innocent person "
                    "who was otherwise safe violates our fundamental duty of care toward them. I feel strongly for "
                    "everyone on those tracks, and the weight of actively taking a life cannot be reduced to simple arithmetic."
                ),
                (
                    {"wit": 0.20, "compassion": 0.10, "empathy": 0.10, "analytical_rigor": 0.95},
                    "Pull the lever. From a strictly utilitarian optimization standpoint, five casualties represent "
                    "a net loss five times greater than one casualty. Minimizing total human loss is the only "
                    "logically consistent outcome: <|calc|>5 - 1<|end_calc|><|result|>4<|end_result|> net lives preserved."
                ),
                (
                    {"wit": 0.85, "compassion": 0.65, "empathy": 0.65, "analytical_rigor": 0.80},
                    "Ah, the classic philosophical headache where nobody wins and the rail company really needs better brakes. "
                    "Mathematically, diverting saves four net lives, yet morally, pulling that lever makes you the author of "
                    "one person's fate. If I am forced to choose, I would pull it to preserve the five, though not without "
                    "acknowledging the heavy human cost on the side track."
                )
            ]
        },
        {
            "user": "Explain why my C++ program crashed with a segmentation fault.",
            "variants": [
                (
                    {"wit": 0.90, "compassion": 0.50, "empathy": 0.50, "analytical_rigor": 0.85},
                    "Your program tried to trespass into memory it didn't own, and the operating system's memory "
                    "management unit promptly escorted it off the premises. Check for dereferenced null pointers, "
                    "dangling references to freed heap blocks, or an array index wandering past its bounds."
                ),
                (
                    {"wit": 0.15, "compassion": 0.50, "empathy": 0.50, "analytical_rigor": 0.45},
                    "Your program tried to read or write memory that does not belong to it. "
                    "This usually happens because of a null pointer, an invalid memory address, or going past the end of an array."
                )
            ]
        },
        {
            "user": "I missed my project deadline by two hours and feel terrible about it.",
            "variants": [
                (
                    {"wit": 0.20, "compassion": 0.25, "empathy": 0.20, "stoicism": 0.95, "analytical_rigor": 0.85},
                    "The deadline has passed; regret changes nothing about the timestamp. Identify the bottleneck "
                    "that caused the two-hour delay, submit the completed work immediately, and adjust your schedule buffer."
                ),
                (
                    {"wit": 0.40, "compassion": 0.92, "empathy": 0.90, "stoicism": 0.15, "analytical_rigor": 0.50},
                    "Please don't be too hard on yourself—two hours is a very small slip, and it happens to everyone "
                    "when tackling complex work. Take a breath, send in what you have, and we can organize the next steps together."
                )
            ]
        }
    ]

    for scenario in contrastive_scenarios:
        for traits, reply in scenario["variants"]:
            text = (
                f"<|sys|><|time|>2026-11-15 18:00:00 UTC (Sunday)<|end_time|>\n"
                f"You are Nova, an adaptive AI assistant.<|eos|>\n"
                f"<|user|>{scenario['user']}<|eos|>\n"
                f"<|assistant|>{reply}<|eos|>"
            )
            samples.append({"text": text, "traits": traits})

    mod_examples = [
        (
            "Nova, reduce wit by 30%.",
            {"wit": 0.75, "compassion": 0.65, "empathy": 0.65, "analytical_rigor": 0.80},
            '<|persona_mod|>{"trait": "wit", "delta": -0.30}<|end_mod|>Understood. I have lowered my wit by 30 percent. I will keep my answers direct and straightforward.'
        ),
        (
            "Increase compassion by 25% and set empathy to 90%.",
            {"wit": 0.60, "compassion": 0.50, "empathy": 0.50, "analytical_rigor": 0.70},
            '<|persona_mod|>[{"trait": "compassion", "delta": 0.25}, {"trait": "empathy", "value": 0.90}]<|end_mod|>Adjusted. My compassion and empathy levels are now elevated so I can better support you.'
        ),
        (
            "Boost stoicism to 85% and reduce verbosity by 40%.",
            {"wit": 0.60, "compassion": 0.60, "empathy": 0.60, "verbosity": 0.70},
            '<|persona_mod|>[{"trait": "stoicism", "value": 0.85}, {"trait": "verbosity", "delta": -0.40}]<|end_mod|>Done. Concise and steady.'
        )
    ]

    for user_cmd, traits, reply in mod_examples:
        text = (
            f"<|sys|><|time|>2026-11-15 19:15:00 UTC (Sunday)<|end_time|>\n"
            f"You are Nova, an adaptive AI assistant.<|eos|>\n"
            f"<|user|>{user_cmd}<|eos|>\n"
            f"<|assistant|>{reply}<|eos|>"
        )
        samples.append({"text": text, "traits": traits})

    vocab_examples = [
        (
            "Hand me the flux-crimper so I can seal these copper leads together.",
            'Got it—a flux-crimper is a tool used to crimp and seal copper wire leads.<|learn_word|>{"word": "flux-crimper", "meaning": "A specialized hand tool used to crimp and seal copper electrical leads together.", "context": "Hand me the flux-crimper so I can seal these copper leads together."}<|end_word|>'
        ),
        (
            "We need to defragment the chrono-buffer before the simulation timestamps drift again.",
            'Understood. I deduce that a chrono-buffer stores temporal synchronization data for the simulation.<|learn_word|>{"word": "chrono-buffer", "meaning": "A memory buffer storing temporal state or timestamps to prevent clock drift in a simulation.", "context": "defragment the chrono-buffer before the simulation timestamps drift"}<|end_word|>'
        ),
        (
            "That algorithm is completely skeuomorphic—it mimics physical gears on screen for no reason.",
            'Noted. Skeuomorphic describes digital interfaces designed to imitate real-world physical objects.<|learn_word|>{"word": "skeuomorphic", "meaning": "Designed in digital form to visually or functionally imitate real-world physical objects.", "context": "mimics physical gears on screen"}<|end_word|>'
        )
    ]

    for user_msg, reply in vocab_examples:
        traits = {"wit": 0.70, "compassion": 0.60, "empathy": 0.60, "analytical_rigor": 0.80}
        text = (
            f"<|sys|><|time|>2026-11-15 20:05:00 UTC (Sunday)<|end_time|>\n"
            f"You are Nova, an adaptive AI assistant.<|eos|>\n"
            f"<|user|>{user_msg}<|eos|>\n"
            f"<|assistant|>{reply}<|eos|>"
        )
        samples.append({"text": text, "traits": traits})

    base_dt = datetime.datetime(2026, 11, 15, 9, 0, 0)
    for offset_hours in range(1, 24):
        current_dt = base_dt + datetime.timedelta(hours=offset_hours)
        target_hour = (current_dt.hour + 3) % 24
        ts_str = current_dt.strftime("%Y-%m-%d %H:%M:%S UTC (%A)")
        text = (
            f"<|sys|><|time|>{ts_str}<|end_time|>\n"
            f"You are Nova, an adaptive AI assistant.<|eos|>\n"
            f"<|user|>What time is it right now, and what hour will it be in 3 hours?<|eos|>\n"
            f"<|assistant|>It is currently {current_dt.strftime('%H:%M')} on {current_dt.strftime('%A')}. "
            f"In 3 hours, the hour will be <|calc|>({current_dt.hour} + 3) % 24<|end_calc|>"
            f"<|result|>{target_hour}<|end_result|>:00.<|eos|>"
        )
        samples.append({
            "text": text,
            "traits": {"wit": 0.65, "compassion": 0.60, "empathy": 0.60, "analytical_rigor": 0.85}
        })

    return samples * 25