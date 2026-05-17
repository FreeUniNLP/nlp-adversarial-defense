from pathlib import Path
from typing import Set

from src.language.entities.word_entity import NounEntry, TagEntry, AxisEntry, VerbEntry, AdjectiveEntry, VerbArgumentEntry, \
    VerbToNounConstraintEntry, AdjectiveToNounConstraint
from src.language.reader import JsonReader

class LexiconParser:

    @staticmethod
    def _parse_axis_min(data: dict) -> AxisEntry:
        """Parse min_axis with permissive defaults for unspecified fields.
        Unspecified fields default to -999 (unconstrained from below)."""
        return AxisEntry(
            agency=data.get("agency", -999),
            physicality=data.get("physicality", -999),
            social=data.get("social", -999),
            system=data.get("system", -999)
        )

    @staticmethod
    def _parse_axis_max(data: dict) -> AxisEntry:
        """Parse max_axis with permissive defaults for unspecified fields.
        Unspecified fields default to 999 (unconstrained from above)."""
        return AxisEntry(
            agency=data.get("agency", 999),
            physicality=data.get("physicality", 999),
            social=data.get("social", 999),
            system=data.get("system", 999)
        )

    @staticmethod
    def parse(path: str | Path):

        raw = JsonReader.read(path)
        nouns = LexiconParser._parse_nouns(raw["NOUNS"])
        verbs = LexiconParser._parse_verbs(raw["VERBS"])
        adjectives = LexiconParser._parse_adjectives(raw["ADJECTIVES"])

        return nouns, verbs, adjectives


    @staticmethod
    def _parse_nouns(data: dict) -> list[NounEntry]:

        entries = []

        for word, values in data.items():
            tags = TagEntry(tag = values["tags"])
            axis = AxisEntry(**values["axis"])

            entry = NounEntry(
                word=word,
                tag=tags,
                axis=axis
            )

            entries.append(entry)

        return entries

    @staticmethod
    def _parse_verbs(data: dict) -> list[VerbEntry]:

        entries = []

        for word, values in data.items():
            tags = TagEntry(tag=values["tags"])
            axis = AxisEntry(**values["axis"])
            arguments = values["arguments"]
            verb_to_subject_constraint =  None
            verb_to_object_constraint = None
            for argument in arguments:
                constraint = argument["constraints"]
                if argument["role"] == "subject":

                    tags_any = TagEntry(tag = constraint["tags_any"])
                    min_axis = LexiconParser._parse_axis_min(constraint["min_axis"])
                    max_axis = LexiconParser._parse_axis_max(constraint["max_axis"])
                    verb_to_subject_constraint = VerbToNounConstraintEntry(
                        tag=tags_any,
                        axis_min=min_axis,
                        axis_max=max_axis
                    )
                else:
                    tags_any = TagEntry(tag=constraint["tags_any"])
                    min_axis = LexiconParser._parse_axis_min(constraint["min_axis"])
                    max_axis = LexiconParser._parse_axis_max(constraint["max_axis"])
                    verb_to_object_constraint = VerbToNounConstraintEntry(
                        tag=tags_any,
                        axis_min=min_axis,
                        axis_max=max_axis
                    )

            verb_argument = VerbArgumentEntry(
                verb_to_subject_constraint=verb_to_subject_constraint,
                verb_to_object_constraint=verb_to_object_constraint
            )

            entry = VerbEntry(
                word=word,
                tag=tags,
                axis=axis,
                verb_argument=verb_argument
            )

            entries.append(entry)

        return entries


    @staticmethod
    def _parse_adjectives(data: dict) -> list[AdjectiveEntry]:

        entries = []

        for word, values in data.items():
            tags = TagEntry(tag=values["tags"])
            axis = AxisEntry(**values["modifies"])
            constraints = values["constraints"]
            tags_any = TagEntry(tag=constraints["tags_any"])
            axis_min = LexiconParser._parse_axis_min(constraints["min_axis"])
            axis_max = LexiconParser._parse_axis_max(constraints["max_axis"])
            adjective_to_noun_constraint = AdjectiveToNounConstraint(
                tag=tags_any,
                axis_min=axis_min,
                axis_max=axis_max
            )
            entry = AdjectiveEntry(
                word=word,
                tag=tags,
                axis=axis,
                adjective_to_noun_constraint=adjective_to_noun_constraint
            )

            entries.append(entry)

        return entries
