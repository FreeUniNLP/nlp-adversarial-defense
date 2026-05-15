from pathlib import Path

from src.language.entities import NounEntry, Axes, VerbEntry, AdjectiveEntry, PartialAxes
from src.language.reader import JsonReader


class LexiconParser:

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

            entry = NounEntry(
                word=word,

                cat1=values["cat1"],
                cat2=values["cat2"],

                axes=Axes(**values["axes"])
            )

            entries.append(entry)

        return entries

    @staticmethod
    def _parse_verbs(data: dict) -> list[VerbEntry]:

        entries = []

        for word, values in data.items():

            entry = VerbEntry(
                word=word,

                cat1=values["cat1"],
                cat2=values["cat2"],

                axes=Axes(**values["axes"]),

                subj_agency_min=values["subj_agency_min"],
                subj_physicality_min=values["subj_physicality_min"],

                obj_cat1=values.get("obj_cat1"),
                obj_physicality_min=values.get("obj_physicality_min")
            )

            entries.append(entry)

        return entries


    @staticmethod
    def _parse_adjectives(data: dict) -> list[AdjectiveEntry]:

        entries = []

        for word, values in data.items():

            entry = AdjectiveEntry(
                word=word,

                cat1=values["cat1"],
                cat2=values["cat2"],

                axes=PartialAxes(**values["axes"]),

                modifies_cat1=values["modifies_cat1"],

                req_physicality_min=values.get("req_physicality_min"),
                req_agency_min=values.get("req_agency_min")
            )

            entries.append(entry)

        return entries