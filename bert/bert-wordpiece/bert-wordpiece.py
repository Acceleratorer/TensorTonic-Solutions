from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        if len(word) > self.max_word_len:
            return [self.unk_token]

        subwords = []
        start = 0

        while start < len(word):
            end = len(word)
            cur_substr = None

            while end > start:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.vocab:
                    cur_substr = substr
                    break

                end -= 1

            if cur_substr is None:
                return [self.unk_token]

            subwords.append(cur_substr)
            start = end

        return subwords