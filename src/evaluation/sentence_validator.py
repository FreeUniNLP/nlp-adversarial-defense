class SentenceValidator:


    def __init__(self):
        pass

    def two_word_validator(self, first_word: str, second_word: str) -> bool:
        ...

    def validate(self, sentence):
        # Basic validation: Check if the sentence is not empty and has at least 3 words
        if not sentence or len(sentence.split()) < 3:
            return False

        # Check for common punctuation errors (e.g., missing period at the end)
        if not sentence.endswith('.'):
            return False

        # Check for capitalization (first letter should be uppercase)
        if not sentence[0].isupper():
            return False

        return True

