from typing import Iterable, Set, Any


class LexiconAnalyzer:
    """Utility service for inspecting and generating statistics over parsed lexical entries."""

    @staticmethod
    def extract_tags(entries: Iterable[Any]) -> Set[str]:
        """Extracts all unique tag strings from any collection of lexical entries."""
        tags = set()
        for entry in entries:
            # Clean duck-typing validation
            tag_wrapper = getattr(entry, 'tag', None)
            if tag_wrapper and hasattr(tag_wrapper, 'tag'):
                tags.update(tag_wrapper.tag)
            elif isinstance(tag_wrapper, list):
                tags.update(tag_wrapper)
        return tags

    @classmethod
    def print_summary(cls, nouns: list, verbs: list, adjectives: list) -> None:
        """Computes and prints tag information profiles to standard output."""
        categories = {
            "NOUN TAGS": cls.extract_tags(nouns),
            "VERB TAGS": cls.extract_tags(verbs),
            "ADJECTIVE TAGS": cls.extract_tags(adjectives)
        }

        for title, tags in categories.items():
            print(f"--- {title} ---")
            print(sorted(list(tags)))
            print(f"Total unique tags: {len(tags)}\n")