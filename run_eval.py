"""
run_eval.py — Evaluate VisVerify on all 10 real-world bbox-CoT samples.

Usage:
    python run_eval.py

Outputs:
  - Per-sample verdict + assert diagnostics
  - Summary table: predicted vs ground truth
  - Accuracy, precision, recall
  - Saved annotated images to ./outputs/
"""

import os
import sys
import json
import time
from PIL import Image, ImageDraw

# ── project imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from dataset import LIMIT, load_sample, load_samples
from registry import EntityRegistry
from visverify import verify_trace

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Annotate image with entity boxes from registry
# -----------------------------------------------------------------------

def annotate(img: Image.Image, registry: EntityRegistry) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    colors = ["#FF0000","#00AA00","#0000FF","#FF8800","#AA00AA","#00AAAA"]
    for i, entity in enumerate(registry.all_entities()):
        b = entity.box
        color = colors[i % len(colors)]
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=3)
        label = f"{entity.entity_id}"
        if entity.attributes:
            attrs = ", ".join(f"{k}={v}" for k, v in entity.attributes.items())
            label += f" [{attrs}]"
        draw.text((b.x1 + 3, b.y1 + 3), label, fill=color)
    return out


# -----------------------------------------------------------------------
# Run evaluation
# -----------------------------------------------------------------------

def run_evaluation():
    results = []
    samples = load_samples(LIMIT)
    total = len(samples)

    print("=" * 70)
    print("  VisVerify Evaluation — 10 Zebra-CoT 3D Counting Samples")
    print("=" * 70)

    for sample in samples:
        sid = sample["id"]
        print(f"\n[{sid:02d}/{total}] {sample['description']}")
        print(f"  Question: {sample['question'][:120]}")
        print(f"  Final answer: {sample['final_answer']}")

        registry = EntityRegistry()

        t0 = time.time()
        try:
            loaded = load_sample(sample)
            img = loaded["image"]
            step_results = verify_trace(img, loaded["question"], loaded["trace"], registry)
            result = step_results[-1]
        except Exception as e:
            import traceback
            print(f"  [ERROR] Exception during verification: {e}")
            traceback.print_exc()
            result = None
            step_results = []
            img = Image.new("RGB", (640, 480), "white")

        elapsed = time.time() - t0

        if result is None:
            predicted = "uncertain"
            correct = False
        else:
            for r in step_results:
                print(r.summary())
            predicted = result.verdict.value.lower()
            gt = sample["ground_truth"]
            correct = predicted == gt

        # Save annotated image
        ann = annotate(img, registry)
        ann_path = os.path.join(OUTPUT_DIR, f"sample_{sid:02d}_annotated.png")
        ann.save(ann_path)

        record = {
            "id": sid,
            "description": sample["description"],
            "question": sample["question"],
            "trace": loaded["trace"] if result else [],
            "final_answer": sample["final_answer"],
            "claim_type": result.claim_type.value if result else "error",
            "ground_truth": sample["ground_truth"],
            "predicted": predicted,
            "confidence": round(result.confidence, 3) if result else 0,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
            "assert_results": (
                [{"name": ar.name, "passed": ar.passed, "message": ar.message}
                 for ar in result.assert_results]
                if result else []
            ),
        }
        results.append(record)

        status_icon = "✓" if correct else "✗"
        print(f"\n  {status_icon} Predicted: {predicted.upper()} | GT: {sample['ground_truth'].upper()} | {elapsed:.1f}s")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'ID':<4} {'Type':<12} {'GT':<6} {'Pred':<10} {'OK':<4} {'Description'}")
    print("-" * 70)
    for r in results:
        icon = "✓" if r["correct"] else "✗"
        print(f"{r['id']:<4} {r['claim_type']:<12} {r['ground_truth']:<6} "
              f"{r['predicted']:<10} {icon}    {r['description'][:40]}")

    correct_total = sum(r["correct"] for r in results)
    accuracy = correct_total / total * 100

    # Precision / Recall on PASS predictions
    tp = sum(1 for r in results if r["predicted"] == "pass" and r["ground_truth"] == "pass")
    fp = sum(1 for r in results if r["predicted"] == "pass" and r["ground_truth"] == "fail")
    fn = sum(1 for r in results if r["predicted"] == "fail" and r["ground_truth"] == "pass")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Accuracy  : {correct_total}/{total} = {accuracy:.1f}%")
    print(f"  Precision : {precision:.2f}  (of predicted PASS, how many were correct)")
    print(f"  Recall    : {recall:.2f}  (of actual PASS, how many did we catch)")
    print(f"  F1        : {f1:.2f}")

    print(f"\n  Annotated images saved to: {OUTPUT_DIR}/")

    # Save JSON results
    results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
