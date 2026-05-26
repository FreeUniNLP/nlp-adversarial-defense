from dataclasses import dataclass, field
from typing import Optional

from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry
from src.language.reader import JsonReader

#TODO

@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        if self.is_valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, error='{self.error}')"


@dataclass
class CFGValidator:
    """
    Validates whether a sentence (as a list of word strings) conforms to:
      1. A valid CFG skeleton (structural match against grammar rules)
      2. Semantic word constraints (tag overlap + cumulative axis bounds)

    Mirrors the generation logic of CFG exactly, but in reverse (parse → validate).
    """

    rules: dict[str, list[list[str]]]

    nouns: list[NounEntry] = field(default_factory=list)
    verbs: list[VerbEntry] = field(default_factory=list)
    adjectives: list[AdjectiveEntry] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def validate(self, sentence: str) -> ValidationResult:
        """
        Full validation pipeline:
          Step 1 – Classify each word token as NOUN / VERB / ADJ
          Step 2 – Check skeleton against CFG rules
          Step 3 – Check semantic constraints (tags + axis)
        """
        tokens = sentence.strip().split()
        if not tokens:
            return ValidationResult(False, "Sentence is empty.")

        # --- Step 1: classify words into abstract skeleton tokens ---
        skeleton, classify_err = self._classify_tokens(tokens)
        if classify_err:
            return ValidationResult(False, classify_err)

        # --- Step 2: validate skeleton against CFG ---
        skeleton_err = self._validate_skeleton(skeleton)
        if skeleton_err:
            return ValidationResult(False, f"Skeleton invalid: {skeleton_err}")

        # --- Step 3: validate semantic constraints ---
        semantic_err = self._validate_semantics(tokens, skeleton)
        if semantic_err:
            return ValidationResult(False, f"Semantic constraint violated: {semantic_err}")

        return ValidationResult(True)

    #  Step 1 – Token classification                                       #
    def _classify_tokens(self, tokens: list[str]) -> tuple[list[str], Optional[str]]:
        """
        Map each surface word to its abstract skeleton label (NOUN / VERB / ADJ).
        Words must exist in exactly one of the three lexicons.
        """
        noun_words  = {n.word: n for n in self.nouns}
        verb_words  = {v.word: v for v in self.verbs}
        adj_words   = {a.word: a for a in self.adjectives}

        skeleton = []
        for word in tokens:
            hits = []
            if word in noun_words: hits.append("NOUN")
            if word in verb_words: hits.append("VERB")
            if word in adj_words:  hits.append("ADJ")

            if len(hits) == 0:
                return [], f"Unknown word '{word}' – not found in nouns, verbs, or adjectives."
            if len(hits) > 1:
                # Ambiguous word – record all possibilities; semantic step will resolve.
                # For skeleton purposes prefer the first hit (priority: NOUN > VERB > ADJ).
                skeleton.append(hits[0])
            else:
                skeleton.append(hits[0])

        return skeleton, None

    #  Step 2 – Skeleton validation                                        #
    def _validate_skeleton(self, skeleton: list[str]) -> Optional[str]:
        """
        Try to parse `skeleton` as a derivation of the start symbol.
        Uses recursive descent over the CFG rules.
        Returns None on success, an error string on failure.
        """
        pos, ok = self._parse_symbol("START", skeleton, 0)
        if not ok:
            return f"Token sequence {skeleton} cannot be derived from START."
        if pos != len(skeleton):
            return (
                f"Parsed only {pos}/{len(skeleton)} tokens. "
                f"Leftover: {skeleton[pos:]}"
            )
        return None

    def _parse_symbol(
        self, symbol: str, skeleton: list[str], pos: int
    ) -> tuple[int, bool]:
        """
        Attempt to match `symbol` starting at `pos`.
        Returns (new_pos, success).
        """
        # Terminal token — match directly
        if symbol not in self.rules:
            if pos < len(skeleton) and skeleton[pos] == symbol:
                return pos + 1, True
            return pos, False

        # Non-terminal — try longest productions first to avoid greedy short-circuit
        # (e.g. VERB_TERM → ["VERB"] would match before ["VERB", "OBJECT", "VERB_TERM"])
        productions = sorted(self.rules[symbol], key=len, reverse=True)
        for production in productions:
            new_pos, ok = self._parse_production(production, skeleton, pos)
            if ok:
                return new_pos, True

        return pos, False

    def _parse_production(
        self, production: list[str], skeleton: list[str], pos: int
    ) -> tuple[int, bool]:
        """Try to match a full production (sequence of symbols) left-to-right."""
        cur = pos
        for sym in production:
            cur, ok = self._parse_symbol(sym, skeleton, cur)
            if not ok:
                return pos, False   # backtrack to original pos
        return cur, True

    # ------------------------------------------------------------------ #
    #  Step 3 – Semantic validation                                        #
    # ------------------------------------------------------------------ #

    def _validate_semantics(
        self, tokens: list[str], skeleton: list[str]
    ) -> Optional[str]:
        """
        Walk the sentence exactly as build_sentence_from_skeleton() does,
        but instead of choosing words we verify the words that are already there.

        Layout (mirrors CFG.build_sentence_from_skeleton):
          [subject block] [verb_chunk]* 
          subject block = (NOUN) | (ADJ NOUN)
          verb_chunk    = VERB | VERB NOUN | VERB ADJ NOUN | VERB ADJ ADJ NOUN
        """
        noun_map = {n.word: n for n in self.nouns}
        verb_map = {v.word: v for v in self.verbs}
        adj_map  = {a.word: a for a in self.adjectives}

        # ---- locate the first VERB to find subject boundary ----
        try:
            first_verb_idx = skeleton.index("VERB")
        except ValueError:
            return "No VERB found in skeleton (should have been caught earlier)."

        subject_skeleton = skeleton[:first_verb_idx]
        subject_tokens   = tokens[:first_verb_idx]

        # ---- validate subject block ----
        subject_adjs: list[AdjectiveEntry] = []
        subject_noun: Optional[NounEntry]  = None

        if subject_skeleton == ["NOUN"]:
            subject_noun = noun_map[subject_tokens[0]]

        elif subject_skeleton == ["ADJ", "NOUN"]:
            adj  = adj_map[subject_tokens[0]]
            noun = noun_map[subject_tokens[1]]
            err = self._check_adj_noun(adj, noun, label="subject")
            if err:
                return err
            subject_adjs = [adj]
            subject_noun = noun

        else:
            return f"Unexpected subject structure (skeleton={subject_skeleton})."

        # ---- split remaining tokens into verb chunks ----
        # A new chunk begins each time we see a VERB token.
        remaining_skeleton = skeleton[first_verb_idx:]
        remaining_tokens   = tokens[first_verb_idx:]

        verb_chunks_sk: list[list[str]] = []
        verb_chunks_tk: list[list[str]] = []
        cur_sk, cur_tk = [], []

        for sk, tk in zip(remaining_skeleton, remaining_tokens):
            if sk == "VERB" and cur_sk:
                verb_chunks_sk.append(cur_sk)
                verb_chunks_tk.append(cur_tk)
                cur_sk, cur_tk = [], []
            cur_sk.append(sk)
            cur_tk.append(tk)

        if cur_sk:
            verb_chunks_sk.append(cur_sk)
            verb_chunks_tk.append(cur_tk)

        # ---- validate each verb chunk ----
        for chunk_sk, chunk_tk in zip(verb_chunks_sk, verb_chunks_tk):
            verb_word = chunk_tk[0]
            verb      = verb_map[verb_word]
            obj_sk    = chunk_sk[1:]
            obj_tk    = chunk_tk[1:]

            # Check verb ↔ subject constraint
            err = self._check_verb_subject(verb, subject_noun, subject_adjs)
            if err:
                return err

            obj_noun_count = obj_sk.count("NOUN")

            # Intransitive verb used with object (or vice-versa)
            if obj_noun_count == 0:
                if verb.verb_argument.verb_to_object_constraint is not None:
                    return (
                        f"Verb '{verb_word}' requires an object "
                        f"but none is present in chunk {chunk_tk}."
                    )
                continue  # nothing more to check for this chunk

            if verb.verb_argument.verb_to_object_constraint is None:
                return (
                    f"Verb '{verb_word}' is intransitive but receives "
                    f"an object in chunk {chunk_tk}."
                )

            v_obj_constraint = verb.verb_argument.verb_to_object_constraint

            # ---- validate object block ----
            if obj_sk == ["NOUN"]:
                obj_noun = noun_map[obj_tk[0]]
                err = self._check_noun_constraint(obj_noun, v_obj_constraint, [], label="object")
                if err:
                    return err

            elif obj_sk == ["ADJ", "NOUN"]:
                obj_adj  = adj_map[obj_tk[0]]
                obj_noun = noun_map[obj_tk[1]]
                err = self._check_adj_noun(obj_adj, obj_noun, label="object adjective")
                if err:
                    return err
                err = self._check_noun_constraint(
                    obj_noun, v_obj_constraint, [obj_adj],
                    label=f"object (under verb '{verb_word}')"
                )
                if err:
                    return err

            elif obj_sk == ["ADJ", "ADJ", "NOUN"]:
                adj1     = adj_map[obj_tk[0]]
                adj2     = adj_map[obj_tk[1]]
                obj_noun = noun_map[obj_tk[2]]

                err = self._check_adj_noun(adj1, obj_noun, label="object adj1")
                if err:
                    return err
                err = self._check_adj_noun(adj2, obj_noun, label="object adj2")
                if err:
                    return err

                # Both adjectives must share at least one tag (overlap check)
                shared = (
                    set(adj1.adjective_to_noun_constraint.tag.tag)
                    & set(adj2.adjective_to_noun_constraint.tag.tag)
                )
                if not shared:
                    return (
                        f"Adjectives '{adj1.word}' and '{adj2.word}' have no overlapping "
                        f"noun-constraint tags and cannot co-modify the same noun."
                    )

                err = self._check_noun_constraint(
                    obj_noun, v_obj_constraint, [adj1, adj2],
                    label=f"object (under verb '{verb_word}', dual adj)"
                )
                if err:
                    return err

            else:
                return f"Unrecognised object structure: {obj_sk}"

        return None  # all checks passed

    # ------------------------------------------------------------------ #
    #  Constraint helpers (mirror CFG._noun_satisfies_constraint)         #
    # ------------------------------------------------------------------ #

    def _noun_satisfies_constraint(
        self,
        noun: NounEntry,
        constraint,
        adjectives: list[AdjectiveEntry] = None,
    ) -> bool:
        """Exact mirror of CFG._noun_satisfies_constraint."""
        if not any(t in noun.tag.tag for t in constraint.tag.tag):
            return False

        agency       = noun.axis.agency
        physicality  = noun.axis.physicality
        social       = noun.axis.social
        system       = noun.axis.system

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

    def _check_noun_constraint(
        self,
        noun: NounEntry,
        constraint,
        adjectives: list[AdjectiveEntry],
        label: str = "",
    ) -> Optional[str]:
        if not self._noun_satisfies_constraint(noun, constraint, adjectives):
            adj_words = [a.word for a in adjectives]
            return (
                f"Noun '{noun.word}' (with adjectives {adj_words}) does not satisfy "
                f"{label} constraint (tag or axis mismatch)."
            )
        return None

    def _check_adj_noun(
        self,
        adj: AdjectiveEntry,
        noun: NounEntry,
        label: str = "",
    ) -> Optional[str]:
        """Check that the adjective's own noun-constraint is met by the noun (without cumulation)."""
        if not self._noun_satisfies_constraint(noun, adj.adjective_to_noun_constraint):
            return (
                f"Adjective '{adj.word}' cannot modify noun '{noun.word}' "
                f"({label}): tag or axis constraint not satisfied."
            )
        return None

    def _check_verb_subject(
        self,
        verb: VerbEntry,
        subject_noun: NounEntry,
        subject_adjs: list[AdjectiveEntry],
    ) -> Optional[str]:
        constraint = verb.verb_argument.verb_to_subject_constraint
        if not self._noun_satisfies_constraint(subject_noun, constraint, subject_adjs):
            adj_words = [a.word for a in subject_adjs]
            return (
                f"Verb '{verb.word}' is incompatible with subject "
                f"'{subject_noun.word}' (adjectives={adj_words}): "
                f"tag or axis constraint not satisfied."
            )
        return None

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cfg(cls, cfg) -> "CFGValidator":
        """Build a validator from an existing CFG instance."""
        return cls(
            rules=cfg.rules,
            nouns=cfg.nouns,
            verbs=cfg.verbs,
            adjectives=cfg.adjectives,
        )

    @classmethod
    def from_json(
        cls,
        file_path: str,
        nouns=None,
        verbs=None,
        adjectives=None,
    ) -> "CFGValidator":
        """Build a validator directly from the same JSON file CFG uses."""
        data = JsonReader.read(file_path)
        return cls(
            rules=data.get("rules", {}),
            nouns=nouns or [],
            verbs=verbs or [],
            adjectives=adjectives or [],
        )