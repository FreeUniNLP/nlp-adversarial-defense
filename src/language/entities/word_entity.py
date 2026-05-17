import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.language.reader import JsonReader


# --- Your Provided Dataclasses ---
@dataclass
class AxisEntry:
    agency: int = 0
    physicality: int = 0
    social: int = 0
    system: int = 0


@dataclass
class TagEntry:
    tag: List[str]


@dataclass
class NounEntry:
    word: str
    tag: TagEntry
    axis: AxisEntry


@dataclass
class VerbToNounConstraintEntry:
    tag: TagEntry
    axis_min: AxisEntry
    axis_max: AxisEntry


@dataclass
class VerbArgumentEntry:
    verb_to_subject_constraint: VerbToNounConstraintEntry
    verb_to_object_constraint: Optional[VerbToNounConstraintEntry]


@dataclass
class VerbEntry:
    word: str
    tag: TagEntry
    axis: AxisEntry
    verb_argument: VerbArgumentEntry


@dataclass
class AdjectiveToNounConstraint:
    tag: TagEntry
    axis_min: AxisEntry
    axis_max: AxisEntry


@dataclass
class AdjectiveEntry:
    word: str
    tag: TagEntry
    axis: AxisEntry
    adjective_to_noun_constraint: AdjectiveToNounConstraint
