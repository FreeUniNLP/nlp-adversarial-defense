from dataclasses import dataclass, field
from typing import Optional

from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry
from src.language.reader import JsonReader


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None

    def __bool__(self): return self.is_valid
    def __repr__(self):
        return "ValidationResult(valid=True)" if self.is_valid \
            else f"ValidationResult(valid=False, error='{self.error}')"


@dataclass
class CFGValidator:
    """
    Validates whether a sentence (as a list of word strings) conforms to:
      1. A valid CFG skeleton (structural match against grammar rules)
      2. Semantic word constraints (tag overlap + cumulative axis bounds)

    Mirrors the generation logic of CFG exactly, but in reverse (parse → validate).
    """

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
    def from_cfg(cls, cfg: "CFG") -> "CFGValidator":
        return cls(rules=cfg.rules, nouns=cfg.nouns, verbs=cfg.verbs, adjectives=cfg.adjectives)

    @classmethod
    def from_json(cls, file_path: str, nouns=None, verbs=None, adjectives=None) -> "CFGValidator":
        data = cls._load_from_json(file_path)
        return cls(rules=data.get("rules", {}), nouns=nouns or [], verbs=verbs or [], adjectives=adjectives or [])