import random
from dataclasses import dataclass, field

from src.language.entities.word_entity import NounEntry, VerbEntry, AdjectiveEntry
from src.language.reader import JsonReader


@dataclass
class CFG:
    start_symbol: str
    rules: dict[str, list[list[str]]] = field(default_factory=dict)

    nouns: list[NounEntry] = field(default_factory=list)
    verbs: list[VerbEntry] = field(default_factory=list)
    adjectives: list[AdjectiveEntry] = field(default_factory=list)

    @classmethod
    def from_json_to_dataclass(cls, file_path: str, nouns=None, verbs=None, adjectives=None) -> 'CFG':
        data = JsonReader.read(file_path)

        start_key = 'start_symbol'
        return cls(
            start_symbol=data.get(start_key),
            rules=data.get("rules", {}),
            nouns=nouns or [],
            verbs=verbs or [],
            adjectives=adjectives or []
        )

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
            context['subject_noun'] = [chosen_noun]
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

            chosen_verb = random.choice(valid_verbs)
            words.append(chosen_verb.word)

            if object_noun_count == 0:
                continue

            v_obj_constraint = chosen_verb.verb_argument.verb_to_object_constraint
            valid_nouns = [n for n in self.nouns if n not in words]

            if object_structure == ['NOUN']:
                valid_nouns_for_verb = [n for n in valid_nouns if self._noun_satisfies_constraint(n, v_obj_constraint)]
                if not valid_nouns_for_verb: raise ValueError(f"No object nouns satisfy verb '{chosen_verb.word}'")
                chosen_noun = random.choice(valid_nouns_for_verb)
                words.append(chosen_noun.word)
            elif object_structure == ['ADJ', 'NOUN']:
                valid_adjs = [
                    a for a in self.adjectives if any(
                        self._noun_satisfies_constraint(n, a.adjective_to_noun_constraint) and
                        self._noun_satisfies_constraint(n, v_obj_constraint, adjectives=[a])
                        for n in valid_nouns
                    )
                ]
                if not valid_adjs: raise ValueError(
                    f"No adjectives fit target constraints under verb '{chosen_verb.word}'")
                chosen_adj = random.choice(valid_adjs)

                final_nouns = [
                    n for n in self.nouns
                    if self._noun_satisfies_constraint(n, chosen_adj.adjective_to_noun_constraint) and
                       self._noun_satisfies_constraint(n, v_obj_constraint, adjectives=[chosen_adj])
                ]
                words.extend([chosen_adj.word, random.choice(final_nouns).word])

            elif object_structure == ['ADJ', 'ADJ', 'NOUN']:
                valid_pairs = []
                for a1 in self.adjectives:
                    for a2 in self.adjectives:
                        if self._do_adjectives_overlap(a1, a2, self.nouns):
                            possible_nouns = [
                                n for n in valid_nouns if
                                self._noun_satisfies_constraint(n, a1.adjective_to_noun_constraint) and
                                self._noun_satisfies_constraint(n, a2.adjective_to_noun_constraint) and
                                self._noun_satisfies_constraint(n, v_obj_constraint, adjectives=[a1, a2])
                            ]
                            if possible_nouns:
                                valid_pairs.append((a1, a2, possible_nouns))

                if not valid_pairs:
                    raise ValueError(
                        f"No overlapping dual adjective pairs fit the requirements for verb '{chosen_verb.word}'")

                chosen_a1, chosen_a2, final_nouns = random.choice(valid_pairs)
                words.extend([chosen_a1.word, chosen_a2.word, random.choice(final_nouns).word])
        return  " ".join(words)

    def _noun_satisfies_constraint(self, noun: NounEntry, constraint, adjectives: list[AdjectiveEntry] = None) -> bool:
        """Returns True if the noun shares a tag and its cumulative axis (Noun + Adjectives) fits the bounds."""
        # 1. Tag Overlap Check (Always run against base noun metadata tags)
        if not any(t in noun.tag.tag for t in constraint.tag.tag):
            return False

        # 2. Extract Base Axis Values
        agency = noun.axis.agency
        physicality = noun.axis.physicality
        social = noun.axis.social
        system = noun.axis.system

        # Cumulative additions: Add modifying adjective weights if present
        if adjectives:
            for adj in adjectives:
                agency += adj.axis.agency
                physicality += adj.axis.physicality
                social += adj.axis.social
                system += adj.axis.system

        # 3. Axis Boundary Check (min <= cumulative_value <= max)
        c_min, c_max = constraint.axis_min, constraint.axis_max
        return (
                c_min.agency <= agency <= c_max.agency and
                c_min.physicality <= physicality <= c_max.physicality and
                c_min.social <= social <= c_max.social and
                c_min.system <= system <= c_max.system
        )

    def _do_adjectives_overlap(self, adj1: AdjectiveEntry, adj2: AdjectiveEntry, nouns_pool: list[NounEntry]) -> bool:
        """Checks if two adjectives share a tag overlap and have at least one valid noun in common."""
        # Tag overlap check between the two constraints
        shared_tags = set(adj1.adjective_to_noun_constraint.tag.tag) & set(adj2.adjective_to_noun_constraint.tag.tag)
        if not shared_tags:
            return False

        # Check if there is at least one noun that satisfies both adjectives
        for noun in nouns_pool:
            if self._noun_satisfies_constraint(noun, adj1.adjective_to_noun_constraint) and \
                    self._noun_satisfies_constraint(noun, adj2.adjective_to_noun_constraint):
                return True

        return False
