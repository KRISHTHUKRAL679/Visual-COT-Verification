"""
Vision backend for VisVerify.

Qwen2.5-VL is the normal perception backend. If img.info["detections"] contains
boxes, those annotations are used instead so small tests can be reproducible.
"""

import json
import os
import re
from functools import lru_cache

from PIL import Image

from registry import BoundingBox


def _norm_to_px(box: dict, w: int, h: int) -> BoundingBox:
    return BoundingBox(
        float(box["x1"]) * w, float(box["y1"]) * h,
        float(box["x2"]) * w, float(box["y2"]) * h,
        float(box.get("confidence", 0.9)),
    )


def _annotated_boxes(img: Image.Image, label: str) -> list[BoundingBox]:
    detections = img.info.get("detections", {})
    boxes = detections.get(label.lower(), [])
    return [_norm_to_px(b, img.width, img.height) for b in boxes]


@lru_cache(maxsize=1)
def _qwen():
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    model_id = os.getenv("VISVERIFY_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    quant = BitsAndBytesConfig(load_in_4bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", quantization_config=quant
    )
    return model, AutoProcessor.from_pretrained(model_id)


def _ask_qwen_json(img: Image.Image, text: str) -> dict:
    model, processor = _qwen()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": text + "\nReturn only JSON."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", raw.strip())
    return json.loads(raw)


def detect_all(img: Image.Image, label: str) -> list[BoundingBox]:
    boxes = _annotated_boxes(img, label)
    if boxes:
        return boxes
    if os.getenv("VISVERIFY_BACKEND", "qwen") == "annotations":
        return []

    prompt = (
        f'Detect every visible "{label}" in the image. '
        'Use normalized coordinates [0,1]. Format: '
        '{"instances":[{"x1":0.1,"y1":0.2,"x2":0.3,"y2":0.4,"confidence":0.9}]}'
    )
    try:
        data = _ask_qwen_json(img, prompt)
        return [_norm_to_px(b, img.width, img.height) for b in data.get("instances", [])]
    except Exception as e:
        print(f"[qwen detect_all] {label}: {e}")
        return []


def detect_primary(img: Image.Image, label: str) -> BoundingBox | None:
    boxes = detect_all(img, label)
    return max(boxes, key=lambda b: b.confidence) if boxes else None


def count_instances(img: Image.Image, label: str) -> int:
    return len(detect_all(img, label))


def classify_color(img: Image.Image, box: BoundingBox) -> str:
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(img.width, int(box.x2)), min(img.height, int(box.y2))
    crop = img.crop((x1, y1, x2, y2)).resize((1, 1))
    r, g, b = crop.getpixel((0, 0))[:3]
    palette = {
        "red": (200, 40, 40), "green": (40, 150, 40), "blue": (50, 80, 190),
        "yellow": (220, 200, 50), "black": (20, 20, 20), "white": (235, 235, 235),
        "gray": (130, 130, 130), "brown": (130, 80, 35), "orange": (220, 120, 30),
    }
    return min(palette, key=lambda c: sum((a - p) ** 2 for a, p in zip((r, g, b), palette[c])))


def classify_attribute(img: Image.Image, box: BoundingBox, attribute: str,
                       choices: list[str] | None = None) -> str:
    return classify_color(img, box) if "color" in attribute.lower() else "unknown"


def is_left_of(a: BoundingBox, b: BoundingBox) -> bool:
    return a.cx < b.cx


def is_right_of(a: BoundingBox, b: BoundingBox) -> bool:
    return a.cx > b.cx


def is_above(a: BoundingBox, b: BoundingBox) -> bool:
    return a.cy < b.cy


def is_below(a: BoundingBox, b: BoundingBox) -> bool:
    return a.cy > b.cy


def is_on_top_of(a: BoundingBox, b: BoundingBox, tolerance_px: float = 30) -> bool:
    return abs(a.y2 - b.y1) < tolerance_px and a.horizontal_overlap_ratio(b) > 0.2


def relative_size(a: BoundingBox, b: BoundingBox) -> str:
    ratio = a.area / b.area if b.area > 0 else 1.0
    return "larger" if ratio > 1.2 else "smaller" if ratio < 0.8 else "similar"


def vqa(img: Image.Image, question: str) -> str:
    if os.getenv("VISVERIFY_BACKEND", "qwen") == "annotations":
        return "unknown"
    try:
        return _ask_qwen_json(img, question).get("answer", "unknown")
    except Exception as e:
        print(f"[qwen vqa] {e}")
        return "unknown"
