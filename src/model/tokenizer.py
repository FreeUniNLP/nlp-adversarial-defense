from __future__ import annotations

from pathlib import Path
from typing import List


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL = [PAD, BOS, EOS, UNK]


class WordTokenizer:
    """Word-level tokenizer built from the mini-language lexicon.

    Vocabulary is built from the words in the corpus file so that every
    token that appears during training is covered.
    """

    def __init__(self, vocab: list[str]):
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token: dict[int, str] = {i: tok for tok, i in self.token_to_id.items()}

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    # ------------------------------------------------------------------ #
    #  Encode / decode                                                     #
    # ------------------------------------------------------------------ #

    def encode(self, sentence: str, add_special: bool = True) -> list[int]:
        """Tokenize a sentence string into a list of token IDs."""
        tokens = sentence.strip().split()
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token IDs back to a sentence string."""
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK)
            if skip_special and i in special_ids:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_corpus(cls, corpus_path: Path) -> "WordTokenizer":
        """Build tokenizer vocabulary from all unique words in a corpus file."""
        words: set[str] = set()
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                words.update(line.strip().split())
        vocab = SPECIAL + sorted(words)
        return cls(vocab)
