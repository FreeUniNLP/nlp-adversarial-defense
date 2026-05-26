import random
from dataclasses import dataclass, field

from src.language.entities.cfg_base import CFGBase
from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry
from src.language.reader import JsonReader


@dataclass
class CFG(CFGBase):
    start_symbol: str

    # Precomputed per-verb lookup tables — built once in __post_init__.
    # _adj1_by_verb[verb_word]  = [(adj, [valid_nouns])]  for ADJ NOUN objects
    # _adj2_by_verb[verb_word]  = [(a1, a2, [valid_nouns])] for ADJ ADJ NOUN objects
    _adj1_by_verb: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _adj2_by_verb: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._build_verb_lookup_tables()

    def _build_verb_lookup_tables(self):
        """Precompute valid (adj, nouns) and (adj1, adj2, nouns) per verb object constraint.
        Runs once at construction; eliminates the O(adj² × nouns) search from every sentence attempt."""
        for verb in self.verbs:
            c = verb.verb_argument.verb_to_object_constraint
            if c is None:
                continue

            # ADJ NOUN objects
            adj1_pairs = []
            for adj in self.adjectives:
                nouns = [
                    n for n in self.nouns
                    if self._noun_satisfies_constraint(n, adj.adjective_to_noun_constraint)
                    and self._noun_satisfies_constraint(n, c, adjectives=[adj])
                ]
                if nouns:
                    adj1_pairs.append((adj, nouns))
            self._adj1_by_verb[verb.word] = adj1_pairs

            # ADJ ADJ NOUN objects
            adj2_pairs = []
            for a1 in self.adjectives:
                shared_tags = set(a1.adjective_to_noun_constraint.tag.tag)
                for a2 in self.adjectives:
                    if not (shared_tags & set(a2.adjective_to_noun_constraint.tag.tag)):
                        continue
                    nouns = [
                        n for n in self.nouns
                        if self._noun_satisfies_constraint(n, a1.adjective_to_noun_constraint)
                        and self._noun_satisfies_constraint(n, a2.adjective_to_noun_constraint)
                        and self._noun_satisfies_constraint(n, c, adjectives=[a1, a2])
                    ]
                    if nouns:
                        adj2_pairs.append((a1, a2, nouns))
            self._adj2_by_verb[verb.word] = adj2_pairs

    def generate_skeleton(self, max_depth: int = 10) -> list[str]:
        return self._expand([self.start_symbol], current_depth=0, max_depth=max_depth)

    def _expand(self, tokens: list[str], current_depth: int, max_depth: int) -> list[str]:
        sentence = []
        for token in tokens:
            if token in self.rules:
                if current_depth >= max_depth:
                    chosen_expansion = min(self.rules[token], key=len)
                else:
                    chosen_expansion = random.choice(self.rules[token])
                sentence.extend(self._expand(chosen_expansion, current_depth + 1, max_depth))
            else:
                sentence.append(token)
        return sentence

    def build_sentence_from_skeleton(self, skeleton_tokens: list[str]) -> str:
        words = []
        context = {}

        first_verb_idx = skeleton_tokens.index('VERB')
        subject_tokens = skeleton_tokens[:first_verb_idx]

        if subject_tokens == ['NOUN']:
            chosen_noun = random.choice(self.nouns)
            words.append(chosen_noun.word)
            context['subject_adjs'] = []
            context['subject_noun'] = chosen_noun
        elif subject_tokens == ['ADJ', 'NOUN']:
            chosen_adj = random.choice(self.adjectives)
            valid_nouns = [n for n in self.nouns if
                           self._noun_satisfies_constraint(n, chosen_adj.adjective_to_noun_constraint)]
            chosen_noun = random.choice(valid_nouns)
            words.extend([chosen_adj.word, chosen_noun.word])
            context['subject_adjs'] = [chosen_adj]
            context['subject_noun'] = chosen_noun
        else:
            raise ValueError(f"Unexpected Subject Structure: {subject_tokens}")

        remaining_tokens = skeleton_tokens[first_verb_idx:]
        verb_chunks = []
        current_chunk = []

        for token in remaining_tokens:
            if token == 'VERB' and current_chunk:
                verb_chunks.append(current_chunk)
                current_chunk = []

            current_chunk.append(token)
        if current_chunk:
            verb_chunks.append(current_chunk)

        for chunk in verb_chunks:
            valid_verbs = [v for v in self.verbs
                           if self._noun_satisfies_constraint(context['subject_noun'],
                                                              v.verb_argument.verb_to_subject_constraint,
                                                              adjectives=context['subject_adjs'])
                           ]

            if not valid_verbs:
                raise ValueError(
                    f"No verbs satisfy the subject properties (cumulative): {context['subject_noun'].word}")

            object_structure = chunk[1:]
            object_noun_count = len([o for o in object_structure if o == 'NOUN'])

            if object_noun_count == 0:
                valid_verbs = [v for v in valid_verbs if v.verb_argument.verb_to_object_constraint is None]
            elif object_noun_count == 1:
                valid_verbs = [v for v in valid_verbs if v.verb_argument.verb_to_object_constraint is not None]
            else:
                raise ValueError(f"Too many object as the argument of verb in {chunk}")

            if not valid_verbs:
                raise ValueError(f"No verbs fit the subject constraints and object structure: {chunk}")

            chosen_verb = random.choice(valid_verbs)
            words.append(chosen_verb.word)

            if object_noun_count == 0:
                continue

            v_obj_constraint = chosen_verb.verb_argument.verb_to_object_constraint
            valid_nouns = [n for n in self.nouns if n.word not in words]

            if object_structure == ['NOUN']:
                valid_nouns_for_verb = [n for n in valid_nouns if self._noun_satisfies_constraint(n, v_obj_constraint)]
                if not valid_nouns_for_verb: raise ValueError(f"No object nouns satisfy verb '{chosen_verb.word}'")
                chosen_noun = random.choice(valid_nouns_for_verb)
                words.append(chosen_noun.word)
            elif object_structure == ['ADJ', 'NOUN']:
                # Use precomputed (adj, nouns) pairs for this verb — just filter out already-used words
                candidates = [
                    (adj, [n for n in nouns if n.word not in words])
                    for adj, nouns in self._adj1_by_verb.get(chosen_verb.word, [])
                ]
                candidates = [(adj, nouns) for adj, nouns in candidates if nouns]
                if not candidates:
                    raise ValueError(f"No adjectives fit target constraints under verb '{chosen_verb.word}'")
                chosen_adj, final_nouns = random.choice(candidates)
                words.extend([chosen_adj.word, random.choice(final_nouns).word])

            elif object_structure == ['ADJ', 'ADJ', 'NOUN']:
                # Use precomputed (a1, a2, nouns) pairs for this verb — just filter out already-used words
                candidates = [
                    (a1, a2, [n for n in nouns if n.word not in words])
                    for a1, a2, nouns in self._adj2_by_verb.get(chosen_verb.word, [])
                ]
                candidates = [(a1, a2, nouns) for a1, a2, nouns in candidates if nouns]
                if not candidates:
                    raise ValueError(
                        f"No overlapping dual adjective pairs fit the requirements for verb '{chosen_verb.word}'")
                chosen_a1, chosen_a2, final_nouns = random.choice(candidates)
                words.extend([chosen_a1.word, chosen_a2.word, random.choice(final_nouns).word])
        return  " ".join(words)

    @classmethod
    def from_json(cls, file_path: str, nouns=None, verbs=None, adjectives=None) -> 'CFG':
        data = JsonReader.read(file_path)

        start_key = 'start_symbol'
        return cls(
            start_symbol=data.get(start_key),
            rules=data.get("rules", {}),
            nouns=nouns or [],
            verbs=verbs or [],
            adjectives=adjectives or []
        )