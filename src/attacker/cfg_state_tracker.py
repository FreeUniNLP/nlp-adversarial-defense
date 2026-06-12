"""
CFGStateTracker — grammar-aware state machine for the attacker.

Tracks which grammatical position we are in at each generation step
and exposes the exact set of words that are valid next tokens.

Grammar (mirrors transition.json):
    START        → SUBJECT_TERM
    SUBJECT_TERM → SUBJECT VERB_TERM
    SUBJECT      → NOUN | ADJ NOUN
    VERB_TERM    → VERB | VERB OBJECT | VERB VERB_TERM | VERB OBJECT VERB_TERM
    OBJECT       → NOUN | ADJ NOUN | ADJ ADJ NOUN

Note on VERB VERB_TERM: a bare verb (used without object — must be
intransitive) can be followed directly by another VERB_TERM, so verb
chains like "MAN RUN FALL" are valid. A transitive verb still requires
its object before the next verb can start.

States follow the sentence left-to-right. At each state the tracker
knows which surface words (not just POS types) are valid, because it
already holds the chosen subject noun/adjs and the current verb, so it
can apply the full semantic constraints (tag overlap + axis bounds).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry


# ------------------------------------------------------------------ #
#  Grammar states                                                      #
# ------------------------------------------------------------------ #

class GrammarState(Enum):
    SUBJECT_START       = auto()   # beginning: expect ADJ or NOUN (subject slot)
    SUBJECT_AFTER_ADJ   = auto()   # after subject ADJ: expect NOUN only
    AFTER_SUBJECT       = auto()   # subject complete: expect VERB
    AFTER_INTRANS_VERB  = auto()   # after intransitive VERB: can end or chain another VERB
    OBJECT_START        = auto()   # after transitive VERB: expect ADJ or NOUN (object slot)
    OBJECT_AFTER_ADJ1   = auto()   # after 1st object ADJ: expect ADJ or NOUN
    OBJECT_AFTER_ADJ2   = auto()   # after 2nd object ADJ: expect NOUN only
    AFTER_OBJECT        = auto()   # object complete: can end or add another VERB


# ------------------------------------------------------------------ #
#  Tracker                                                             #
# ------------------------------------------------------------------ #

class CFGStateTracker:
    """
    Stateful grammar tracker.  Call reset() before each new prefix.

    Usage:
        tracker = CFGStateTracker(nouns, verbs, adjectives)
        tracker.reset()
        words, can_end = tracker.valid_next_words()
        ok = tracker.step("MAN")   # advance state with chosen word
    """

    def __init__(
        self,
        nouns: list[NounEntry],
        verbs: list[VerbEntry],
        adjectives: list[AdjectiveEntry],
    ) -> None:
        self.nouns      = nouns
        self.verbs      = verbs
        self.adjectives = adjectives

        # Fast lookup maps
        self._noun_map = {n.word: n for n in nouns}
        self._verb_map = {v.word: v for v in verbs}
        self._adj_map  = {a.word: a for a in adjectives}

        self._precompute_viability()
        self.reset()

    # ------------------------------------------------------------------ #
    #  Winnability precomputation                                          #
    # ------------------------------------------------------------------ #

    def _precompute_viability(self) -> None:
        """Precompute which words keep the sentence completable.

        Guarantees that every prefix the tracker allows (including every
        point where can_end=True) admits at least one valid full-sentence
        completion. Without this, the attacker can emit dead-end prefixes
        like a bare subject that no verb is compatible with — a prefix the
        defender can never complete validly.

          * safe verb     — intransitive, or has at least one valid object
          * viable noun   — has at least one safe verb compatible with it
                            (as subject, given its adjectives)
          * viable adj    — leads to at least one viable noun
        """
        # --- safe verbs ---
        self._safe_verbs: list[VerbEntry] = []
        for v in self.verbs:
            c = v.verb_argument.verb_to_object_constraint
            if c is None:
                self._safe_verbs.append(v)
                continue
            has_bare_obj = any(self._noun_ok(n, c) for n in self.nouns)
            if has_bare_obj or any(
                self._noun_ok(n, a.adjective_to_noun_constraint)
                and self._noun_ok(n, c, [a])
                for a in self.adjectives for n in self.nouns
            ):
                self._safe_verbs.append(v)

        def has_safe_verb(noun: NounEntry, adjs: list[AdjectiveEntry]) -> bool:
            return any(
                self._noun_ok(noun, v.verb_argument.verb_to_subject_constraint, adjs)
                for v in self._safe_verbs
            )

        # --- viable subject nouns (bare, and per leading adjective) ---
        self._viable_bare_nouns: list[str] = [
            n.word for n in self.nouns if has_safe_verb(n, [])
        ]
        self._viable_nouns_after_adj: dict[str, list[str]] = {}
        for a in self.adjectives:
            ns = [
                n.word for n in self.nouns
                if self._noun_ok(n, a.adjective_to_noun_constraint)
                and has_safe_verb(n, [a])
            ]
            if ns:
                self._viable_nouns_after_adj[a.word] = ns
        self._viable_subject_adjs: list[str] = list(self._viable_nouns_after_adj.keys())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset to the initial state (start of a new sentence)."""
        self.state: GrammarState  = GrammarState.SUBJECT_START
        self.subject_noun: Optional[NounEntry]       = None
        self.subject_adjs: list[AdjectiveEntry]      = []
        self.current_verb: Optional[VerbEntry]       = None
        self.object_adjs:  list[AdjectiveEntry]      = []
        self.generated:    list[str]                 = []

    @property
    def can_end(self) -> bool:
        """True if the current prefix is a valid stopping point."""
        return (
            len(self.generated) > 0
            and self.state not in (GrammarState.SUBJECT_START,)
        )

    def sentence(self) -> str:
        """Return the prefix generated so far as a space-joined string."""
        return " ".join(self.generated)

    def valid_next_words(self) -> tuple[list[str], bool]:
        """
        Return (valid_words, can_end).

        valid_words  — surface words (uppercase strings) allowed as the next token.
        can_end      — whether EOS is also a legal action at this step.
        """
        s = self.state

        if s == GrammarState.SUBJECT_START:
            # Only viable words: every offered subject keeps the sentence
            # completable (has at least one safe verb available).
            words = self._viable_subject_adjs + self._viable_bare_nouns
            return words, False   # can't end before writing anything

        if s == GrammarState.SUBJECT_AFTER_ADJ:
            # Must place a NOUN compatible with the subject adjective AND
            # viable (some safe verb accepts adj+noun as subject).
            adj = self.subject_adjs[-1]
            words = list(self._viable_nouns_after_adj.get(adj.word, []))
            return words, True    # "BIG" alone is a valid prefix

        if s == GrammarState.AFTER_SUBJECT:
            words = self._valid_verbs_for_subject()
            return words, True    # "MAN" alone is a valid prefix

        if s == GrammarState.AFTER_INTRANS_VERB:
            # Grammar: VERB_TERM → VERB VERB_TERM
            # A bare intransitive verb can be followed by another VERB_TERM,
            # so any verb valid for the subject may chain here — or we end.
            words = self._valid_verbs_for_subject()
            return words, True

        if s == GrammarState.OBJECT_START:
            words = self._valid_object_starts()
            return words, True    # prefix can end before the object

        if s == GrammarState.OBJECT_AFTER_ADJ1:
            words = self._valid_after_obj_adj1()
            return words, True

        if s == GrammarState.OBJECT_AFTER_ADJ2:
            words = self._valid_after_obj_adj2()
            return words, True

        if s == GrammarState.AFTER_OBJECT:
            # Can continue with another VERB, or stop
            words = self._valid_verbs_for_subject()
            return words, True

        return [], True

    def step(self, word: str) -> bool:
        """
        Advance the grammar state by placing `word`.
        Returns True on success, False if `word` is not valid here.
        """
        valid_words, _ = self.valid_next_words()
        if word not in valid_words:
            return False

        self._advance(word)
        self.generated.append(word)
        return True

    # ------------------------------------------------------------------ #
    #  State transition logic                                              #
    # ------------------------------------------------------------------ #

    def _advance(self, word: str) -> None:
        s = self.state

        if s == GrammarState.SUBJECT_START:
            if word in self._adj_map:
                self.subject_adjs.append(self._adj_map[word])
                self.state = GrammarState.SUBJECT_AFTER_ADJ
            else:
                self.subject_noun = self._noun_map[word]
                self.state = GrammarState.AFTER_SUBJECT

        elif s == GrammarState.SUBJECT_AFTER_ADJ:
            self.subject_noun = self._noun_map[word]
            self.state = GrammarState.AFTER_SUBJECT

        elif s in (GrammarState.AFTER_SUBJECT,
                   GrammarState.AFTER_INTRANS_VERB,
                   GrammarState.AFTER_OBJECT):
            verb = self._verb_map[word]
            self.current_verb = verb
            self.object_adjs  = []
            if verb.verb_argument.verb_to_object_constraint is None:
                self.state = GrammarState.AFTER_INTRANS_VERB
            else:
                self.state = GrammarState.OBJECT_START

        elif s == GrammarState.OBJECT_START:
            if word in self._adj_map:
                self.object_adjs = [self._adj_map[word]]
                self.state = GrammarState.OBJECT_AFTER_ADJ1
            else:
                self.object_adjs = []
                self.state = GrammarState.AFTER_OBJECT

        elif s == GrammarState.OBJECT_AFTER_ADJ1:
            if word in self._adj_map:
                self.object_adjs.append(self._adj_map[word])
                self.state = GrammarState.OBJECT_AFTER_ADJ2
            else:
                self.object_adjs = []
                self.state = GrammarState.AFTER_OBJECT

        elif s == GrammarState.OBJECT_AFTER_ADJ2:
            self.object_adjs = []
            self.state = GrammarState.AFTER_OBJECT

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _valid_verbs_for_subject(self) -> list[str]:
        """Safe verbs whose subject constraint is satisfied by the current subject.

        Restricted to safe verbs (intransitive or with at least one valid
        object) so a prefix ending right after a transitive verb is always
        completable.
        """
        return [
            v.word for v in self._safe_verbs
            if self._noun_ok(
                self.subject_noun,
                v.verb_argument.verb_to_subject_constraint,
                self.subject_adjs,
            )
        ]

    def _valid_object_starts(self) -> list[str]:
        """
        First tokens of a valid OBJECT given the current transitive verb.
        Includes both bare nouns and adjectives that lead to at least one valid noun.
        """
        c = self.current_verb.verb_argument.verb_to_object_constraint

        valid_nouns = [
            n.word for n in self.nouns
            if self._noun_ok(n, c)
        ]
        valid_adjs = [
            a.word for a in self.adjectives
            if any(
                self._noun_ok(n, a.adjective_to_noun_constraint)
                and self._noun_ok(n, c, [a])
                for n in self.nouns
            )
        ]
        return valid_nouns + valid_adjs

    def _valid_after_obj_adj1(self) -> list[str]:
        """Valid tokens after the first object adjective (NOUN or second ADJ)."""
        c    = self.current_verb.verb_argument.verb_to_object_constraint
        adj1 = self.object_adjs[0]

        valid_nouns = [
            n.word for n in self.nouns
            if self._noun_ok(n, adj1.adjective_to_noun_constraint)
            and self._noun_ok(n, c, [adj1])
        ]

        valid_adjs = []
        for adj2 in self.adjectives:
            # adj2 must share at least one tag with adj1
            shared = (
                set(adj1.adjective_to_noun_constraint.tag.tag)
                & set(adj2.adjective_to_noun_constraint.tag.tag)
            )
            if not shared:
                continue
            # must have at least one noun satisfying both adjs + verb constraint
            if any(
                self._noun_ok(n, adj1.adjective_to_noun_constraint)
                and self._noun_ok(n, adj2.adjective_to_noun_constraint)
                and self._noun_ok(n, c, [adj1, adj2])
                for n in self.nouns
            ):
                valid_adjs.append(adj2.word)

        return valid_nouns + valid_adjs

    def _valid_after_obj_adj2(self) -> list[str]:
        """Valid NOUN tokens after two object adjectives."""
        c    = self.current_verb.verb_argument.verb_to_object_constraint
        adj1 = self.object_adjs[0]
        adj2 = self.object_adjs[1]

        return [
            n.word for n in self.nouns
            if self._noun_ok(n, adj1.adjective_to_noun_constraint)
            and self._noun_ok(n, adj2.adjective_to_noun_constraint)
            and self._noun_ok(n, c, [adj1, adj2])
        ]

    def _noun_ok(
        self,
        noun: NounEntry,
        constraint,
        adjectives: list[AdjectiveEntry] = None,
    ) -> bool:
        """
        Mirror of CFGBase._noun_satisfies_constraint.
        True if noun+adjectives satisfy the given constraint (tag + axis).
        """
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
