# cfg_base.py
from dataclasses import dataclass, field
from typing import Optional
from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry
from src.language.reader import JsonReader

@dataclass(kw_only=True)
class CFGBase:
    """
    Shared foundation for CFG generation and validation.
    Owns the grammar data and the single implementation of constraint checking.
    """
    rules: dict[str, list[list[str]]]
    nouns: list[NounEntry] = field(default_factory=list)
    verbs: list[VerbEntry] = field(default_factory=list)
    adjectives: list[AdjectiveEntry] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Core constraint logic — single source of truth                     #
    # ------------------------------------------------------------------ #

    def _noun_satisfies_constraint(
        self,
        noun: NounEntry,
        constraint,
        adjectives: list[AdjectiveEntry] = None,
    ) -> bool:
        if not any(t in noun.tag.tag for t in constraint.tag.tag):
            return False

        agency      = noun.axis.agency
        physicality = noun.axis.physicality
        social      = noun.axis.social
        system      = noun.axis.system

        if adjectives:
            for adj in adjectives:
                agency      += adj.axis.agency
                physicality += adj.axis.physicality
                social      += adj.axis.social
                system      += adj.axis.system

        c_min, c_max = constraint.axis_min, constraint.axis_max
        return (
            c_min.agency      <= agency      <= c_max.agency      and
            c_min.physicality <= physicality <= c_max.physicality and
            c_min.social      <= social      <= c_max.social      and
            c_min.system      <= system      <= c_max.system
        )

    # ------------------------------------------------------------------ #
    #  Skeleton parsing — used by both generator and validator            #
    # ------------------------------------------------------------------ #

    def _parse_symbol(self, symbol: str, skeleton: list[str], pos: int) -> tuple[int, bool]:
        if symbol not in self.rules:
            if pos < len(skeleton) and skeleton[pos] == symbol:
                return pos + 1, True
            return pos, False

        for production in sorted(self.rules[symbol], key=len, reverse=True):
            new_pos, ok = self._parse_production(production, skeleton, pos)
            if ok:
                return new_pos, True
        return pos, False

    def _parse_production(self, production: list[str], skeleton: list[str], pos: int) -> tuple[int, bool]:
        cur = pos
        for sym in production:
            cur, ok = self._parse_symbol(sym, skeleton, cur)
            if not ok:
                return pos, False
        return cur, True

    def _validate_skeleton(self, skeleton: list[str]) -> Optional[str]:
        pos, ok = self._parse_symbol("START", skeleton, 0)
        if not ok:
            return f"Token sequence {skeleton} cannot be derived from START."
        if pos != len(skeleton):
            return f"Parsed only {pos}/{len(skeleton)} tokens. Leftover: {skeleton[pos:]}"
        return None

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_from_json(cls, file_path: str) -> dict:
        """Raw JSON load — subclasses call this in their own from_json()."""
        return JsonReader.read(file_path)