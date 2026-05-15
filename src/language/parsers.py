from pathlib import Path

from src.language.entities import NounEntry, TagEntry, AxesEntry, VerbEntry, AdjectiveEntry, VerbArgumentEntry, \
    VerbToNounConstraintEntry
from src.language.reader import JsonReader


class LexiconParser:

    @staticmethod
    def parse(path: str | Path):

        raw = JsonReader.read(path)
        print(raw)
        nouns = LexiconParser._parse_nouns(raw["NOUNS"])
        verbs = LexiconParser._parse_verbs(raw["VERBS"])
        adjectives = LexiconParser._parse_adjectives(raw["ADJECTIVES"])

        return nouns, verbs, adjectives


    @staticmethod
    def _parse_nouns(data: dict) -> list[NounEntry]:

        entries = []

        for word, values in data.items():
            tags = TagEntry(tag = values["tags"])
            axes = AxesEntry(**values["axes"])

            entry = NounEntry(
                word=word,
                tag=tags,
                axes=axes
            )

            entries.append(entry)

        return entries

    @staticmethod
    def _parse_verbs(data: dict) -> list[VerbEntry]:

        entries = []

        for word, values in data.items():
            tags = TagEntry(tag=values["tags"])
            axes = AxesEntry(**values["axes"])
            arguments = values["arguments"]
            # only subject

            for argument in arguments:
                if argument["role"] == "subject":
                    constraint = argument["constraints"]
                    tags_any = TagEntry(tag = constraint["tags_any"])
                    min_axis = AxesEntry(**values["min_axis"])
                    max_axis = AxesEntry(**values["max_axes"])
                    verb_to_subject_constraint = VerbToNounConstraintEntry(
                        tag=tags_any,
                        axes_min=min_axis,
                        axes_max=max_axis
                    )


            #constarits
            entry = VerbEntry(
                word=word,
            )

            entries.append(entry)

        return entries


    @staticmethod
    def _parse_adjectives(data: dict) -> list[AdjectiveEntry]:

        entries = []

        for word, values in data.items():
            tags = TagEntry(tag=values["tags"])
            axes = AxesEntry(physicality=values["modifies"])
            #constraints
            entry = AdjectiveEntry(
            )

            entries.append(entry)

        return entries
