import json
from pathlib import Path


class JsonReader:

    @staticmethod
    def read(path: str | Path) -> dict:

        path = Path(path)

        with open(path, "r") as file:
            return json.load(file)