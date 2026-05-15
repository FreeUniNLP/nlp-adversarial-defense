from dataclasses import dataclass
from typing import Optional

@dataclass
class Axes:
    agency: int = 0
    physicality: int = 0
    social: int = 0
    system: int = 0

@dataclass
class NounEntry:
    cat1: str
    cat2: str
    axes: Axes


@dataclass
class VerbEntry:
    cat1: str
    cat2: str
    axes: Axes

    subj_agency_min: int
    subj_physicality_min: int

    obj_cat1: Optional[list[str]] = None
    obj_physicality_min: Optional[int] = None

@dataclass
class AdjectiveEntry:
    cat1: str
    cat2: str

    axes: dict

    modifies_cat1: list[str]

    req_physicality_min: Optional[int] = None
    req_agency_min: Optional[int] = None