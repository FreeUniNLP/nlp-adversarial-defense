from dataclasses import dataclass
from typing import Optional

@dataclass
class axisEntry:
    agency: int = 0
    physicality: int = 0
    social: int = 0
    system: int = 0

@dataclass
class TagEntry:
    tag: list[str]

@dataclass
class NounEntry:
    word: str
    tag: TagEntry
    axis: axisEntry

@dataclass
class VerbToNounConstraintEntry:
    tag: TagEntry
    axis_min: axisEntry
    axis_max: axisEntry

@dataclass
class VerbArgumentEntry:
    verb_to_subject_constraint: VerbToNounConstraintEntry
    verb_to_object_constraint: Optional[VerbToNounConstraintEntry]

@dataclass
class VerbEntry:
    word: str
    tag: TagEntry
    axis: axisEntry
    verb_argument: VerbArgumentEntry

@dataclass
class AdjectiveToNounConstraint:
    tag: TagEntry
    axis_min: axisEntry
    axis_max: axisEntry

@dataclass
class AdjectiveEntry:
    word: str
    tag: TagEntry
    axis: axisEntry
    adjective_to_noun_constraint: AdjectiveToNounConstraint

