"""
VisVerify — Core Algorithm

VisVerify(I, Q, C[1..i-1], s_i)
  1. CLAIM PARSE     → typed claim
  2. CONTEXT EXTRACT → reuse EntityRegistry
  3. ENTITY LOCALIZE → ground new entities
  4. PROGRAM SYNTH   → generate typed verification program
  5. EXECUTE         → run asserts, collect evidence, emit verdict
"""

import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from PIL import Image

from registry import EntityRegistry, Entity, BoundingBox
import vision_tools as vt


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ClaimType(Enum):
    EXISTENCE   = "existence"
    COUNTING    = "counting"
    SPATIAL     = "spatial"
    ATTRIBUTE   = "attribute"
    COMPARISON  = "comparison"
    RELATIONAL  = "relational"


class Verdict(Enum):
    PASS      = "PASS"
    FAIL      = "FAIL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class AssertResult:
    name: str
    passed: bool
    message: str
    evidence: dict = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class VerificationResult:
    step_index: int
    claim: str
    claim_type: ClaimType
    verdict: Verdict
    assert_results: list[AssertResult]
    confidence: float = 0.0
    contradiction_flags: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Step {self.step_index}: [{self.verdict.value}] {self.claim}",
            f"Claim type: {self.claim_type.value}",
        ]
        for ar in self.assert_results:
            icon = "✓" if ar.passed else "✗"
            lines.append(f"  {icon} {ar.name} p={ar.confidence:.2f}: {ar.message}")
            if ar.evidence:
                for k, v in ar.evidence.items():
                    lines.append(f"      evidence.{k} = {v}")
        if self.contradiction_flags:
            lines.append("  ⚠ CONTRADICTIONS:")
            for c in self.contradiction_flags:
                lines.append(f"    {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claim Parser and trace fact extractor
# ---------------------------------------------------------------------------

def parse_claim(claim: str, context_steps: list[str]) -> tuple[ClaimType, dict]:
    text = claim.lower()
    meta = {"entities": [], "predicate": "", "expected_value": None, "count": None, "boxes": []}

    for b in re.findall(r"\[([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\]", text):
        meta["boxes"].append([float(x) for x in b])

    count = re.search(r"(?:there (?:is|are)|i see|see)\s+(?:exactly\s+)?(\d+)\s+([a-z ]+?)(?:\s+in|\s+on|\.|$)", text)
    if count:
        meta.update(count=int(count.group(1)), entities=[_clean_label(count.group(2))])
        return ClaimType.COUNTING, meta

    visible = re.search(r"(?:the\s+)?([a-z0-9 ]+?)\s+at\s+\[[^\]]+\]\s+(?:is|are)\s+visible", text)
    if visible:
        meta["entities"] = [_clean_label(visible.group(1))]
        return ClaimType.EXISTENCE, meta

    spatial = re.search(r"(?:the\s+)?([a-z0-9 ]+?)\s+(?:is|are)?\s*(?:to\s+the\s+)?(left|right|above|below|on top of)\s+(?:of\s+)?(?:the\s+)?([a-z0-9 ]+?)(?:\.|$)", text)
    if spatial:
        pred = spatial.group(2).replace(" ", "_")
        meta.update(entities=[_clean_label(spatial.group(1)), _clean_label(spatial.group(3))], predicate=pred)
        return ClaimType.SPATIAL, meta

    comp = re.search(r"([a-z0-9 ]+?)\s+is\s+(larger|bigger|smaller|taller|shorter)\s+than\s+([a-z0-9 ]+)", text)
    if comp:
        meta.update(entities=[_clean_label(comp.group(1)), _clean_label(comp.group(3))], predicate=comp.group(2))
        return ClaimType.COMPARISON, meta

    attr = re.search(r"(?:the\s+)?([a-z0-9 ]+?)\s+is\s+(red|green|blue|yellow|black|white|gray|brown|orange)", text)
    if attr:
        meta.update(entities=[_clean_label(attr.group(1))], predicate="color", expected_value=attr.group(2))
        return ClaimType.ATTRIBUTE, meta

    exist = re.search(r"there (?:is|are)\s+(?:a|an|the)?\s*([a-z0-9 ]+?)(?:\s+in|\s+on|\.|$)", text)
    if exist:
        meta["entities"] = [_clean_label(exist.group(1))]
        return ClaimType.EXISTENCE, meta

    return ClaimType.RELATIONAL, meta


def _clean_label(text: str) -> str:
    text = re.sub(r"\b(exactly|visible|image|scene|is|are)\b", "", text)
    label = " ".join(text.split()).strip()
    label = re.sub(r"^(the|a|an)\s+", "", label)
    if label.endswith("ches"):
        label = label[:-2]
    elif label.endswith("ies"):
        label = label[:-3] + "y"
    elif label.endswith("s") and not label.endswith("ss"):
        label = label[:-1]
    return label


def extract_claimed_facts(step: str, prior_steps: list[str], step_index: int,
                          registry: EntityRegistry) -> list[dict]:
    claim_type, meta = parse_claim(step, prior_steps)
    facts = []
    if claim_type == ClaimType.COUNTING:
        facts.append({"key": f"count:{meta['entities'][0]}", "value": meta["count"]})
    elif claim_type == ClaimType.EXISTENCE and meta["entities"]:
        facts.append({"key": f"exists:{meta['entities'][0]}", "value": True})
    elif claim_type == ClaimType.ATTRIBUTE:
        facts.append({"key": f"attr:{meta['entities'][0]}:{meta['predicate']}", "value": meta["expected_value"]})
    elif claim_type in {ClaimType.SPATIAL, ClaimType.COMPARISON} and len(meta["entities"]) >= 2:
        facts.append({"key": f"{claim_type.value}:{meta['entities'][0]}:{meta['predicate']}:{meta['entities'][1]}", "value": True})
    if meta.get("boxes"):
        facts.append({"key": f"boxes:{','.join(meta.get('entities') or ['unknown'])}", "value": meta["boxes"]})
    for fact in facts:
        registry.add_claimed_fact(fact["key"], fact["value"], 1.0, step_index)
    return facts


# ---------------------------------------------------------------------------
# Verification Program Templates
# ---------------------------------------------------------------------------

def _run_asserts(fns: list[tuple[str, Callable]]) -> list[AssertResult]:
    """
    Execute a list of (name, fn) pairs.
    fn() should return (bool, message, evidence_dict).
    Stops at first failure — earlier failure explains later failures.
    """
    results = []
    for name, fn in fns:
        try:
            passed, message, evidence = fn()
            conf = float(evidence.get("confidence", 1.0 if passed else 0.0))
            results.append(AssertResult(name, passed, message, evidence, conf))
            if not passed:
                break  # diagnostic stack: stop at first failure
        except Exception as e:
            results.append(AssertResult(
                name, False,
                f"Exception during check: {e}",
                {"traceback": traceback.format_exc(limit=2)},
                0.0,
            ))
            break
    return results


def verify_existence(img: Image.Image, claim: str, meta: dict,
                     registry: EntityRegistry, step: int) -> list[AssertResult]:
    entities = meta.get("entities", [])
    if not entities:
        return [AssertResult("parse", False, "Could not extract entity from claim", {})]

    target = entities[0]

    def check_grounding():
        # Check registry first
        existing = registry.get_all(target)
        if existing:
            return True, f"'{target}' already in registry from prior step", \
                   {"entity_id": existing[0].entity_id, "confidence": existing[0].box.confidence}
        # Fresh detection
        boxes = vt.detect_all(img, target)
        if boxes:
            box = max(boxes, key=lambda b: b.confidence)
            e = registry.register(target, box, step=step)
            return True, f"Detected '{target}' at {boxes[0]}", \
                   {"entity_id": e.entity_id, "box": str(box), "count_found": len(boxes), "confidence": box.confidence}
        return False, f"'{target}' not found in image", {"count_found": 0, "confidence": 0.0}

    return _run_asserts([("existence_check", check_grounding)])


def verify_counting(img: Image.Image, claim: str, meta: dict,
                    registry: EntityRegistry, step: int) -> list[AssertResult]:
    entities = meta.get("entities", [])
    expected_count = meta.get("count")
    if not entities or expected_count is None:
        return [AssertResult("parse", False, "Could not parse count claim", {})]

    target = entities[0]

    def check_detection():
        boxes = vt.detect_all(img, target)
        if not boxes:
            return False, f"No '{target}' detected at all", {"found": 0, "confidence": 0.0}
        if not registry.get_all(target):
            for b in boxes:
                registry.register(target, b, step=step)
        conf = _count_confidence(len(boxes), expected_count, boxes)
        return True, f"Found {len(boxes)} instance(s) of '{target}'", {"found": len(boxes), "confidence": conf}

    def check_count():
        boxes = vt.detect_all(img, target)
        found = len(boxes)
        conf = _count_confidence(found, expected_count, boxes)
        passed = found == expected_count
        msg = (f"Count matches: {found} == {expected_count}" if passed
               else f"Count mismatch: found {found}, expected {expected_count}")
        return passed, msg, {"found": found, "expected": expected_count, "confidence": conf}

    return _run_asserts([
        ("detection_precondition", check_detection),
        ("count_assertion",        check_count),
    ])


def verify_spatial(img: Image.Image, claim: str, meta: dict,
                   registry: EntityRegistry, step: int) -> list[AssertResult]:
    entities = meta.get("entities", [])
    predicate = meta.get("predicate", "")
    if len(entities) < 2:
        return [AssertResult("parse", False,
                             "Spatial claim needs 2 entities", {})]

    subj_label, ref_label = entities[0], entities[1]

    def localize_subject():
        e = registry.get_first(subj_label) or _localize_and_register(
            img, subj_label, registry, step)
        if e is None:
            return False, f"Cannot localize subject '{subj_label}'", {}
        return True, f"Subject '{subj_label}' at {e.box}", {"id": e.entity_id, "confidence": e.box.confidence}

    def localize_reference():
        e = registry.get_first(ref_label) or _localize_and_register(
            img, ref_label, registry, step)
        if e is None:
            return False, f"Cannot localize reference '{ref_label}'", {}
        return True, f"Reference '{ref_label}' at {e.box}", {"id": e.entity_id, "confidence": e.box.confidence}

    def check_spatial():
        subj = registry.get_first(subj_label)
        ref  = registry.get_first(ref_label)
        if not subj or not ref:
            return False, "Entity missing after localization", {}

        sb, rb = subj.box, ref.box
        pred_lower = predicate.lower()

        spatial_checks = {
            "left":      (vt.is_left_of(sb, rb),  f"cx({subj_label})={sb.cx:.0f} vs cx({ref_label})={rb.cx:.0f}"),
            "right":     (vt.is_right_of(sb, rb), f"cx({subj_label})={sb.cx:.0f} vs cx({ref_label})={rb.cx:.0f}"),
            "above":     (vt.is_above(sb, rb),    f"cy({subj_label})={sb.cy:.0f} vs cy({ref_label})={rb.cy:.0f}"),
            "below":     (vt.is_below(sb, rb),    f"cy({subj_label})={sb.cy:.0f} vs cy({ref_label})={rb.cy:.0f}"),
            "on_top_of": (vt.is_on_top_of(sb, rb), f"y2({subj_label})={sb.y2:.0f} vs y1({ref_label})={rb.y1:.0f}"),
        }

        for key, (result, detail) in spatial_checks.items():
            if key in pred_lower:
                msg = (f"Spatial '{key}' holds: {detail}" if result
                       else f"Spatial '{key}' FAILED: {detail}")
                conf = min(sb.confidence, rb.confidence) * (0.9 if result else 0.2)
                return result, msg, {"predicate": key, "detail": detail, "confidence": conf}

        answer = vt.vqa(img, f"Is the {subj_label} {predicate} the {ref_label}? Answer yes or no.")
        passed = "yes" in answer.lower()
        return passed, f"Qwen VQA: '{answer}'", {"vqa_answer": answer, "confidence": 0.5 if passed else 0.2}

    return _run_asserts([
        ("localize_subject",   localize_subject),
        ("localize_reference", localize_reference),
        ("spatial_assertion",  check_spatial),
    ])


def verify_attribute(img: Image.Image, claim: str, meta: dict,
                     registry: EntityRegistry, step: int) -> list[AssertResult]:
    entities = meta.get("entities", [])
    predicate = meta.get("predicate", "")
    expected  = meta.get("expected_value", "")
    if not entities:
        return [AssertResult("parse", False, "No entity in attribute claim", {})]

    target = entities[0]

    def localize():
        e = registry.get_first(target) or _localize_and_register(
            img, target, registry, step)
        if e is None:
            return False, f"Cannot localize '{target}'", {}
        return True, f"'{target}' at {e.box}", {"confidence": e.box.confidence}

    def check_attr():
        entity = registry.get_first(target)
        if not entity:
            return False, "Entity not in registry", {}

        # Color is common enough to have a dedicated tool
        if "color" in predicate.lower() or "colour" in predicate.lower():
            actual = vt.classify_color(img, entity.box)
        else:
            actual = vt.classify_attribute(img, entity.box, predicate)

        registry.update_attribute(entity.entity_id, predicate, actual, step=step)

        if expected:
            passed = expected.lower() in actual.lower() or actual.lower() in expected.lower()
            msg = (f"Attribute '{predicate}'='{actual}' matches expected '{expected}'" if passed
                   else f"Attribute '{predicate}'='{actual}', expected '{expected}'")
        else:
            passed = actual != "unknown"
            msg = f"Attribute '{predicate}'='{actual}'"

        conf = entity.box.confidence * (0.85 if passed else 0.25)
        return passed, msg, {"attribute": predicate, "actual": actual, "expected": expected, "confidence": conf}

    return _run_asserts([
        ("localize_entity",    localize),
        ("attribute_assertion", check_attr),
    ])


def verify_comparison(img: Image.Image, claim: str, meta: dict,
                      registry: EntityRegistry, step: int) -> list[AssertResult]:
    entities = meta.get("entities", [])
    predicate = meta.get("predicate", "larger")
    if len(entities) < 2:
        return [AssertResult("parse", False, "Comparison needs 2 entities", {})]

    e1_label, e2_label = entities[0], entities[1]

    def localize_both():
        e1 = registry.get_first(e1_label) or _localize_and_register(img, e1_label, registry, step)
        e2 = registry.get_first(e2_label) or _localize_and_register(img, e2_label, registry, step)
        if not e1:
            return False, f"Cannot localize '{e1_label}'", {}
        if not e2:
            return False, f"Cannot localize '{e2_label}'", {}
        return True, f"Both entities localized", {"confidence": min(e1.box.confidence, e2.box.confidence)}

    def check_comparison():
        e1 = registry.get_first(e1_label)
        e2 = registry.get_first(e2_label)
        if not e1 or not e2:
            return False, "Missing entity", {}
        rel = vt.relative_size(e1.box, e2.box)
        pred_lower = predicate.lower()
        if "larger" in pred_lower or "bigger" in pred_lower:
            passed = rel == "larger"
        elif "smaller" in pred_lower:
            passed = rel == "smaller"
        else:
            passed = True  # can't assess
        msg = (f"Size comparison: {e1_label}({e1.box.area:.0f}px²) is '{rel}' vs "
               f"{e2_label}({e2.box.area:.0f}px²)")
        return passed, msg, {
            "area_e1": e1.box.area, "area_e2": e2.box.area,
            "relation": rel, "expected": predicate,
            "confidence": min(e1.box.confidence, e2.box.confidence) * (0.9 if passed else 0.2),
        }

    return _run_asserts([
        ("localize_both",       localize_both),
        ("comparison_assertion", check_comparison),
    ])


def verify_relational(img: Image.Image, claim: str, meta: dict,
                      registry: EntityRegistry, step: int) -> list[AssertResult]:
    """Use Qwen VQA for complex relational claims."""
    def check_via_vqa():
        answer = vt.vqa(img, f"In this image, is it true that: {claim}? Answer yes or no.")
        passed = "yes" in answer.lower()
        return passed, f"VQA: '{answer}' for claim '{claim}'", {"vqa_answer": answer, "confidence": 0.5 if passed else 0.2}

    return _run_asserts([("relational_vqa", check_via_vqa)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _localize_and_register(img: Image.Image, label: str,
                            registry: EntityRegistry, step: int) -> Entity | None:
    box = vt.detect_primary(img, label)
    if box is None:
        return None
    return registry.register(label, box, step=step)


def _count_confidence(found: int, expected: int, boxes: list[BoundingBox]) -> float:
    if not boxes:
        return 0.0
    mean_det = sum(b.confidence for b in boxes) / len(boxes)
    count_score = 1.0 if found == expected else 1.0 / (1.0 + abs(found - expected))
    return mean_det * count_score


def _anchor_claim_boxes(img: Image.Image, meta: dict, registry: EntityRegistry, step: int) -> None:
    labels, boxes = meta.get("entities", []), meta.get("boxes", [])
    for label, b in zip(labels, boxes):
        if registry.get_first(label):
            continue
        claimed = BoundingBox(
            b[0] * img.width, b[1] * img.height,
            b[2] * img.width, b[3] * img.height,
            confidence=0.2,
        )
        detected = vt.detect_all(img, label)
        best = max(detected, key=lambda d: _iou(claimed, d), default=None)
        if best and _iou(claimed, best) >= 0.5:
            registry.register(label, best, step=step)
        else:
            registry.register(label, claimed, step=step)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _verdict_from_conf(assert_results: list[AssertResult]) -> tuple[Verdict, float]:
    if not assert_results:
        return Verdict.UNCERTAIN, 0.0
    conf = min(ar.confidence for ar in assert_results)
    if all(ar.passed for ar in assert_results) and conf >= CONFIDENCE_THRESHOLD:
        return Verdict.PASS, conf
    if conf < CONFIDENCE_THRESHOLD:
        return Verdict.UNCERTAIN, conf
    return Verdict.FAIL, conf


VERIFIER_MAP: dict[ClaimType, Callable] = {
    ClaimType.EXISTENCE:  verify_existence,
    ClaimType.COUNTING:   verify_counting,
    ClaimType.SPATIAL:    verify_spatial,
    ClaimType.ATTRIBUTE:  verify_attribute,
    ClaimType.COMPARISON: verify_comparison,
    ClaimType.RELATIONAL: verify_relational,
}


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.4


def visverify(
    img: Image.Image,
    question: str,
    context_steps: list[str],
    current_claim: str,
    step_index: int,
    registry: EntityRegistry,
) -> VerificationResult:
    """
    VisVerify(I, Q, C[1..i-1], s_i)

    Args:
        img           : the image
        question      : the original question Q
        context_steps : C[1..i-1] — prior verified CoT steps
        current_claim : s_i — the claim to verify now
        step_index    : i (1-based)
        registry      : shared EntityRegistry (mutated in place)

    Returns:
        VerificationResult with verdict, assert diagnostics, evidence
    """
    for j, prior in enumerate(context_steps, start=1):
        extract_claimed_facts(prior, context_steps[:j - 1], j, registry)

    print(f"\n[VisVerify] Step {step_index}: parsing claim...")
    extract_claimed_facts(current_claim, context_steps, step_index, registry)
    claim_type, meta = parse_claim(current_claim, context_steps)
    _anchor_claim_boxes(img, meta, registry, step_index)
    print(f"[VisVerify] Claim type: {claim_type.value} | meta: {meta}")

    verifier = VERIFIER_MAP.get(claim_type, verify_relational)
    assert_results = verifier(img, current_claim, meta, registry, step_index)

    verdict, local_conf = _verdict_from_conf(assert_results)
    chain_conf = registry.step_confidence.get(step_index - 1, 1.0)
    confidence = min(local_conf, chain_conf)
    registry.step_confidence[step_index] = confidence

    return VerificationResult(
        step_index=step_index,
        claim=current_claim,
        claim_type=claim_type,
        verdict=verdict,
        assert_results=assert_results,
        confidence=confidence,
        contradiction_flags=list(registry.contradiction_log),
    )


def verify_trace(img: Image.Image, question: str, trace: list[str],
                 registry: EntityRegistry | None = None) -> list[VerificationResult]:
    registry = registry or EntityRegistry()
    results = []
    for i, step in enumerate(trace, start=1):
        result = visverify(img, question, trace[:i - 1], step, i, registry)
        results.append(result)
        if result.verdict == Verdict.FAIL:
            break
    return results
