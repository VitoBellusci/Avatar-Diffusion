import json
from typing import List, Dict


# E' importante isolare i set per evitare il data leakage
# config contiene gli iperparametri per centralizzare la pipeline di pre processing

class AvatarTokenizer:
    def __init__(self, config=None):
        self.config = config
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def fit(self, training_texts: List[str]):
        """
        Costruisce il vocabolario, partendo esclusivamente dalle parole presenti nel training set.
        A qualsiasi parola trovata, per esempio in fase di inferenza, viene assegnata il pad <UNK>
        """
        for text in training_texts:
            # Divide le parole del testo e le converte in minuscolo
            tokens = text.lower().split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Converte una stringa in una sequenza di interi
        """
        tokens = text.lower().split()
        encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Padding o Truncation
        if len(encoded) < self.config.max_seq_len:
            encoded += [self.vocab["<PAD>"]] * (self.config.max_seq_len - len(encoded))
        else:
            encoded = encoded[:self.config.max_seq_len]

        return encoded

    def decode(self, ids: List[int]) -> str:
        return " ".join([self.inverse_vocab.get(i, "<UNK>") for i in ids])

    # Salva il vocabolario in json   
    def save_vocab(self):
        with open(self.config.vocab_path, 'w') as f:
            json.dump(self.vocab, f)

    # Carica un vocabolario precedentemente salvato
    def load_vocab(self):
        with open(self.config.vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
