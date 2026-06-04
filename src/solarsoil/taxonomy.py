"""Label taxonomy for solar-panel condition.

The original project collapsed several source categories into a binary
clean/dirty label. The published version keeps a richer, physically-motivated
taxonomy and lets each experiment choose a *label space*:

* ``binary``      — clean vs dirty (the original task).
* ``condition``   — clean / soiled / damaged  (3-way, the headline framing).
* ``multiclass``  — the 6 canonical source classes.

Why the split between *soiling* and *damage*? They are physically different
problems: soiling (dust, bird droppings, snow) is removable and causes a
temporary power loss, whereas damage (physical, electrical) is permanent and
needs repair. Modelling them as one flat label hides that distinction, so we
expose a 2-level hierarchy (condition -> type).

Note on "debris": the user's original idea included a *debris* class, but the
source datasets do not label debris separately — it is visually folded into
``Dusty``. We keep the 6 canonical classes and document this in DATASET.md.
"""
from __future__ import annotations

# The six canonical classes as labelled by the Kaggle "Faulty solar panel"
# dataset (pythonafroz). Folder names there use these spellings.
MULTICLASS: list[str] = [
    "Clean",
    "Dusty",
    "Bird-drop",
    "Snow-Covered",
    "Physical-Damage",
    "Electrical-damage",
]

# Level-1 condition labels.
CONDITION: list[str] = ["clean", "soiled", "damaged"]

# Binary labels (original task). ImageFolder sorts alphabetically -> clean=0.
BINARY: list[str] = ["clean", "dirty"]

# type -> condition (level-2 -> level-1)
TYPE_TO_CONDITION: dict[str, str] = {
    "Clean": "clean",
    "Dusty": "soiled",
    "Bird-drop": "soiled",
    "Snow-Covered": "soiled",
    "Physical-Damage": "damaged",
    "Electrical-damage": "damaged",
}

# type -> binary (clean vs everything else)
TYPE_TO_BINARY: dict[str, str] = {
    cls: ("clean" if cls == "Clean" else "dirty") for cls in MULTICLASS
}

# Map a raw source folder name (any case / spelling variant) to a canonical
# multiclass label. Extend here when a new dataset uses different folder names.
_ALIASES: dict[str, str] = {
    "clean": "Clean",
    "dusty": "Dusty",
    "dust": "Dusty",
    "dirty": "Dusty",
    "bird-drop": "Bird-drop",
    "bird_drop": "Bird-drop",
    "birddrop": "Bird-drop",
    "snow-covered": "Snow-Covered",
    "snow_covered": "Snow-Covered",
    "snow": "Snow-Covered",
    "physical-damage": "Physical-Damage",
    "physical_damage": "Physical-Damage",
    "electrical-damage": "Electrical-damage",
    "electrical_damage": "Electrical-damage",
}


def normalize_label(raw: str) -> str:
    """Map a raw folder/label string to a canonical multiclass label."""
    key = raw.strip().lower().replace(" ", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    key2 = key.replace("-", "_")
    if key2 in _ALIASES:
        return _ALIASES[key2]
    raise KeyError(f"Unknown label {raw!r}; add it to taxonomy._ALIASES")


def label_space(name: str) -> list[str]:
    """Return the ordered class list for a label space name."""
    return {"binary": BINARY, "condition": CONDITION, "multiclass": MULTICLASS}[name]


def to_label_space(canonical: str, space: str) -> str:
    """Project a canonical multiclass label into ``space``."""
    if space == "multiclass":
        return canonical
    if space == "condition":
        return TYPE_TO_CONDITION[canonical]
    if space == "binary":
        return TYPE_TO_BINARY[canonical]
    raise ValueError(f"Unknown label space {space!r}")
