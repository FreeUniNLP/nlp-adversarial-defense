from dataclasses import dataclass
from typing import Optional

@dataclass
class AxesEntry:
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
    axes: AxesEntry

@dataclass
class VerbToNounConstraintEntry:
    tag: TagEntry
    axes_min: AxesEntry
    axes_max: AxesEntry

@dataclass
class VerbArgumentEntry:
    verb_to_subject_constraint: VerbToNounConstraintEntry
    verb_to_object_constraint: Optional[VerbToNounConstraintEntry]

@dataclass
class VerbEntry:
    tag: TagEntry
    axes: AxesEntry
    verb_argument: VerbArgumentEntry

@dataclass
class AdjectiveToNounConstraint:
    tag: TagEntry
    axes_min: AxesEntry
    axes_max: AxesEntry

@dataclass
class AdjectiveEntry:
    tag: TagEntry
    axes: AxesEntry
    adjective_to_noun_constraint: AdjectiveToNounConstraint

