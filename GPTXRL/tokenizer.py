import numpy as np
import json

class Tokenizer:
    def __init__(self) -> None:
        self.vocab = []
        self.input_size = 12

    def load_from_file(self, path:str):
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.loads(f.read())['vocab']

    def tokenize(self, string, should_continue=False):
        characters = string.split()
        tokens = []

        for char in characters:
            try:
                tokens.append(self.vocab.index(char))
            except:
                tokens.append(0)

        while len(tokens) < self.input_size and should_continue == False:
            tokens.append(0)

        return np.array(tokens, dtype=np.int32)

    def detokenize(self, tokens):
        detokenized = []
        for token in tokens:
            detokenized.append(self.vocab[token])
        return ' '.join(detokenized)