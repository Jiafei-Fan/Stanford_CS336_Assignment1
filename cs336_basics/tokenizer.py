import json
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.byte_to_int = {b: i for i, b in vocab.items()}
    
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        # vocab.json: token_str (latin-1) -> id
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id: dict[str, int] = json.load(f)

        # build id -> bytes
        vocab: dict[int, bytes] = {idx: tok.encode("latin-1") for tok, idx in token_to_id.items()}

        merges: list[tuple[bytes, bytes]] = []
        # merges.txt: each line "a_str b_str" where str is latin-1 representation of raw bytes
        with open(merges_path, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.rstrip("\n")
                if not line:
                    continue
                char_list:list[str] = [i for i in line]
                if char_list[0] == " ":
                    for i in range(1, len(char_list)):
                        if char_list[i] == " ":
                            a_str = "".join(char_list[0:i])
                            b_str = "".join(char_list[i+1:])
                            break
                else:
                    a_str, b_str = line.split(" ", 1)
                merges.append((a_str.encode("latin-1"), b_str.encode("latin-1")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        parts: list[str] = []
        # make sure to split out special tokens first
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            delim = "|".join(re.escape(s) for s in specials)
            parts = re.split(f"({delim})", text)
        else:
            parts = [text]
        
        out_ids: list[int] = []

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in parts:
            if part == "":
                continue

            # if part itself is a special token, search directly in vocab
            if part in self.special_tokens:
                b = part.encode("utf-8")
                id = self.byte_to_int[b]
                out_ids.append(id)
                continue

            # if not, then we pre-tokenize
            for m in re.finditer(PAT, part):
                pretok = m.group(0)

                bs = pretok.encode("utf-8")
                tokens: list[bytes] = [bytes([x]) for x in bs] # [b"t", b"h", b"e"]
                tokens = self._apply_bpe(tokens)

                # map tokens to ids
                for t in tokens:
                    id = next(k for k,v in self.vocab.items() if v == t)
                    out_ids.append(id)

        return out_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into the original text.
        """
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

    def _apply_bpe(self, tokens: list[bytes]) -> list[bytes]:
        # [b't', b'h', b'e']
        while True:
            if len(tokens) < 2:
                return tokens
            # merge adjacent pairs with lowest rank first
            best_i = None
            best_rank = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair) # preventing a KeyError if the key is not found
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_i = i

            # no merge found
            if best_i is None:
                return tokens

            # merge best_i and best_i + 1
            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2:]            
