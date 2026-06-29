"""
Entity Registry — progressive scene graph built across CoT steps.
Stores grounded entities, their bounding boxes, attributes, and relations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    def horizontal_overlap_ratio(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_x2 = min(self.x2, other.x2)
        if inter_x2 <= inter_x1:
            return 0.0
        inter_w = inter_x2 - inter_x1
        min_w = min(self.width, other.width)
        return inter_w / min_w if min_w > 0 else 0.0

    def vertical_overlap_ratio(self, other: "BoundingBox") -> float:
        inter_y1 = max(self.y1, other.y1)
        inter_y2 = min(self.y2, other.y2)
        if inter_y2 <= inter_y1:
            return 0.0
        inter_h = inter_y2 - inter_y1
        min_h = min(self.height, other.height)
        return inter_h / min_h if min_h > 0 else 0.0

    def __repr__(self):
        return f"BBox([{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}] conf={self.confidence:.2f})"


@dataclass
class Entity:
    entity_id: str              # e.g. "person_1", "car_2"
    label: str                  # class label
    box: BoundingBox
    attributes: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)   # {entity_id: relation_str}
    step_introduced: int = 0    # which CoT step first grounded this entity

    def __repr__(self):
        return f"Entity({self.entity_id}, {self.box}, attrs={self.attributes})"


@dataclass
class Fact:
    key: str
    value: object
    confidence: float
    step: int
    source: str = "trace"


class EntityRegistry:
    """
    Running scene graph. Populated incrementally as CoT steps are verified.
    Supports contradiction detection across steps.
    """

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._label_counters: dict[str, int] = {}
        self.contradiction_log: list[str] = []
        self.facts: dict[str, Fact] = {}
        self.step_confidence: dict[int, float] = {0: 1.0}

    def register(self, label: str, box: BoundingBox, attrs: dict = None,
                 step: int = 0) -> Entity:
        """Add a newly grounded entity; returns its Entity object."""
        self._label_counters[label] = self._label_counters.get(label, 0) + 1
        entity_id = f"{label}_{self._label_counters[label]}"
        entity = Entity(
            entity_id=entity_id,
            label=label,
            box=box,
            attributes=attrs or {},
            step_introduced=step,
        )
        self._entities[entity_id] = entity
        self.facts[f"exists:{label}"] = Fact(
            key=f"exists:{label}", value=True, confidence=box.confidence,
            step=step, source="vision"
        )
        return entity

    def get(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_all(self, label: str) -> list[Entity]:
        return [e for e in self._entities.values() if e.label == label]

    def get_first(self, label: str) -> Optional[Entity]:
        matches = self.get_all(label)
        return matches[0] if matches else None

    def update_attribute(self, entity_id: str, key: str, value,
                         step: int = 0) -> None:
        """Update an attribute, logging contradictions if value changes."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return
        if key in entity.attributes and entity.attributes[key] != value:
            msg = (
                f"[CONTRADICTION] Step {step}: entity '{entity_id}' attr '{key}' "
                f"was '{entity.attributes[key]}' (step {entity.step_introduced}), "
                f"now '{value}'"
            )
            self.contradiction_log.append(msg)
        entity.attributes[key] = value
        self.facts[f"attr:{entity.label}:{key}"] = Fact(
            key=f"attr:{entity.label}:{key}", value=value,
            confidence=entity.box.confidence, step=step, source="vision"
        )

    def add_claimed_fact(self, key: str, value, confidence: float,
                         step: int, source: str = "trace") -> None:
        old = self.facts.get(key)
        if old and old.value != value:
            self.contradiction_log.append(
                f"[CONTRADICTION] Step {step}: fact '{key}' was '{old.value}', now '{value}'"
            )
        self.facts[key] = Fact(key, value, confidence, step, source)

    def get_fact(self, key: str) -> Optional[Fact]:
        return self.facts.get(key)

    def add_relation(self, entity_id: str, other_id: str, relation: str) -> None:
        entity = self._entities.get(entity_id)
        if entity:
            entity.relations[other_id] = relation

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def __repr__(self):
        lines = ["EntityRegistry:"]
        for e in self._entities.values():
            lines.append(f"  {e}")
        if self.contradiction_log:
            lines.append("  CONTRADICTIONS:")
            for c in self.contradiction_log:
                lines.append(f"    {c}")
        if self.facts:
            lines.append("  FACTS:")
            for f in self.facts.values():
                lines.append(f"    {f.key}={f.value} p={f.confidence:.2f} ({f.source})")
        return "\n".join(lines)
