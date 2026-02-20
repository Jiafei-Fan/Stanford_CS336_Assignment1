from cs336_basics.train_bpe import train_bpe
import time
import json

def train_bpe_owt():
    input_path = "./data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.3f} seconds")

    # serialize vocab and merges
    json_file_path = "./data/owt_bpe_vocab.json"
    merges_file_path = "./data/owt_bpe_merges.txt"
    # vocab: dict[int, bytes]  -> dict[str, int]
    vocab_out = {b.decode("latin-1"): i for i, b in vocab.items()}
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(vocab_out, json_file, ensure_ascii=False, indent=4)

    with open(merges_file_path, "w", encoding="utf-8") as merges_file:
        for a, b in merges:
            merges_file.write(f"{a.decode('latin-1')} {b.decode('latin-1')}\n")

    # longest_token = max(vocab.values(), key=len)
    # print(longest_token)

if __name__ == "__main__":
    train_bpe_owt()
    
    