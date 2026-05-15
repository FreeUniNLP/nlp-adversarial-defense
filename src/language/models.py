from dataclasses import dataclass

@dataclass
class AxesEntry:
    agency: int
    physicality: int
    social: int
    system: int

@dataclass
class VerbSlotsEntry:
    ...