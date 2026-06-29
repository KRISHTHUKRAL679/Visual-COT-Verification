"""
Zebra-CoT loader.

Streams only 10 samples from:
  multimodal-reasoning-lab/Zebra-CoT
  3D Visual Reasoning - Multi-Hop Objects Counting / train

Cached fields: Question, Text Reasoning Trace, Final Answer, problem_image_1.
"""

import json
import re
from pathlib import Path

from PIL import Image

DATASET_ID = "multimodal-reasoning-lab/Zebra-CoT"
CONFIG = "3D Visual Reasoning - Multi-Hop Objects Counting"
SPLIT = "train"
LIMIT = 10

CACHE_DIR = Path(__file__).with_name("data") / "zebra_3d_counting_10"
META_PATH = CACHE_DIR / "samples.jsonl"


def _clean_text(text: str) -> str:
    return re.sub(r"<image_start>.*?<image_end>", "", str(text), flags=re.S).strip()


def _trace_steps(trace: str) -> list[str]:
    trace = _clean_text(trace)
    parts = re.split(r"\bTHOUGHT\s+\d+:\s*", trace)
    return [p.strip() for p in parts if p.strip()]


def _save_image(img, path: Path) -> None:
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img.convert("RGB").save(path)


def prepare_zebra_samples(limit: int = LIMIT) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if META_PATH.exists():
        return _read_cached(limit)

    from datasets import load_dataset

    rows = load_dataset(DATASET_ID, CONFIG, split=SPLIT, streaming=True)
    samples = []
    with META_PATH.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows.take(limit), start=1):
            image_path = CACHE_DIR / f"sample_{i:02d}.jpg"
            _save_image(row["problem_image_1"], image_path)
            sample = {
                "id": i,
                "question": _clean_text(row["Question"]),
                "text_reasoning_trace": _clean_text(row["Text Reasoning Trace"]),
                "final_answer": str(row["Final Answer"]).strip(),
                "image_path": str(image_path),
                "ground_truth": "pass",
                "description": f"Zebra-CoT 3D counting sample {i}",
            }
            f.write(json.dumps(sample) + "\n")
            samples.append(sample)
    return samples


def _read_cached(limit: int = LIMIT) -> list[dict]:
    with META_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for _, line in zip(range(limit), f)]


def load_samples(limit: int = LIMIT) -> list[dict]:
    return prepare_zebra_samples(limit)


def load_sample(sample: dict) -> dict:
    img = Image.open(sample["image_path"]).convert("RGB")
    trace = _trace_steps(sample["text_reasoning_trace"])
    trace.append(
        f'For the question "{sample["question"]}", the final answer is "{sample["final_answer"]}".'
    )
    return {
        **sample,
        "image": img,
        "trace": trace,
    }
