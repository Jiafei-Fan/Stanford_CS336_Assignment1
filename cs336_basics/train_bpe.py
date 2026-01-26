import time


def train_bpe_tinystories():
    from tests.adapters import run_train_bpe
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    print("Vocabulary:", vocab)
    print("Merges:", merges)


if __name__ == "__main__":
    start_time = time.time()
    train_bpe_tinystories()
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.3f} seconds")
