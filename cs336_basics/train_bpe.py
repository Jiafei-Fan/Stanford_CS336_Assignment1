import time
import cProfile
import os
import multiprocessing
from collections import Counter
import regex as re
from typing import List, Tuple
from .pretokenization_example import find_chunk_boundaries

def _pretokenize(
    text: str,
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    """Pretokenize the given text string.
    Split on special tokens, so that no merging can occur across the text they delimit.

    Args:
        text (str): Input text to pre-tokenize.

    Returns:
        dict[tuple[bytes], int]: A dictionary mapping from token byte tuples to their counts.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    dict_token_counts: dict[tuple[bytes], int] = {}
    if not special_tokens:
        raise AssertionError("special_tokens must be a non-empty list of strings")
    delim = "|".join(re.escape(tok) for tok in special_tokens)
    # split a string to list of substrings, also remove the delimiters such as <|endoftext|>
    text_parts = re.split(delim, text)
    # avoid considering last string in DOC1 and first string in DOC2 as adjacent
    for part in text_parts:
        for match in re.finditer(PAT, part):
            token_bytes_sequence = match.group(0).encode("utf-8")
            # Store token counts for BPE training
            token_bytes_tuple: tuple[bytes, ...] = tuple(bytes([b]) for b in token_bytes_sequence)
            if token_bytes_tuple in dict_token_counts:
                dict_token_counts[token_bytes_tuple] += 1
            else:
                dict_token_counts[token_bytes_tuple] = 1
    return dict_token_counts

def _count_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return Counter(_pretokenize(chunk, special_tokens))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes), utf-8 encoded.
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Initialize vocab with 256 and special tokens
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    special_token_id = 256
    for special_token in special_tokens:
        vocab[special_token_id] = special_token.encode("utf-8")
        special_token_id += 1

    with open(input_path, "rb") as f:
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        global_counts: Counter[tuple[bytes], int] = Counter()
        chunk_args = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_counters = pool.map(_count_chunk, chunk_args)
            for counter in chunk_counters:
                global_counts.update(counter)
            
        # Compute BPE merges from global_counts
        merges: list[tuple[bytes, bytes]] = []
        pair_counts: Counter[tuple[bytes, bytes], int] = Counter()
        firsttime = True
        while len(vocab) < vocab_size:
            # pair_counts.clear()
            if firsttime:
                for token_tuple, count in global_counts.items():
                    # Get all adjacent pairs in token_tuple
                    for i in range(len(token_tuple) - 1):
                        pair = (token_tuple[i], token_tuple[i + 1])
                        pair_counts[pair] += count
                firsttime = False
            else:
                for old_tu, new_tu, cnt in updated_tuples_set:
                    # subtract old pairs
                    for j in range(len(old_tu) - 1):
                        p = (old_tu[j], old_tu[j + 1])
                        newv = pair_counts[p] - cnt
                        if newv:
                            pair_counts[p] = newv
                        else:
                            del pair_counts[p]

                    # add new pairs
                    for j in range(len(new_tu) - 1):
                        p = (new_tu[j], new_tu[j + 1])
                        pair_counts[p] += cnt
            max_count = max(pair_counts.values())
            ties: list[tuple[bytes, bytes]] = [pair for pair, c in pair_counts.items() if c == max_count]
            lexicographically_max_pair: tuple[bytes, bytes] = max(ties)
            merges.append(lexicographically_max_pair)
            # update vocab with new merged token
            vocab[len(vocab)] = lexicographically_max_pair[0] + lexicographically_max_pair[1]
            # Update global_counts by merging the selected pair
            merged_first, merged_second = lexicographically_max_pair
            merged_token: bytes = merged_first + merged_second
            updated_counts: Counter[tuple[bytes], int] = Counter()
            updated_tuples_set: set[tuple[tuple[bytes, ...], tuple[bytes, ...], int]] = set()

            for token_tuple, count in global_counts.items():
                merged_sequence: list[bytes] = []
                i = 0
                modified = False
                while i < len(token_tuple):    
                    if (
                        i + 1 < len(token_tuple)
                        and token_tuple[i] == merged_first
                        and token_tuple[i + 1] == merged_second
                    ):
                        merged_sequence.append(merged_token)
                        i += 2
                        modified = True
                    else:
                        merged_sequence.append(token_tuple[i])
                        i += 1
                new_tuple = tuple(merged_sequence)
                updated_counts[new_tuple] += count
                if modified:
                    updated_tuples_set.add((token_tuple, new_tuple, count))
            global_counts = updated_counts

        return vocab, merges



