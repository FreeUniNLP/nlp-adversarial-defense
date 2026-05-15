from dataclasses import dataclass
from typing import Optional


# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class Axes:
    agency: int = 0
    physicality: int = 0
    social: int = 0
    system: int = 0


@dataclass
class PartialAxes:
    agency: Optional[int] = None
    physicality: Optional[int] = None
    social: Optional[int] = None
    system: Optional[int] = None


@dataclass
class NounEntry:
    word: str

    cat1: str
    cat2: str

    axes: Axes


@dataclass
class VerbEntry:
    word: str

    cat1: str
    cat2: str

    axes: Axes

    subj_agency_min: int
    subj_physicality_min: int

    obj_cat1: Optional[list[str]] = None
    obj_physicality_min: Optional[int] = None


@dataclass
class AdjectiveEntry:
    word: str

    cat1: str
    cat2: str

    axes: PartialAxes

    modifies_cat1: list[str]

    req_physicality_min: Optional[int] = None
    req_agency_min: Optional[int] = None