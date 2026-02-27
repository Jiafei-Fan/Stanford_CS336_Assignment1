import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input/target sequences for next-token prediction.

    Args:
        x: 1D numpy array of token ids.
        batch_size: number of sequences in the batch.
        context_length: length of each sequence.
        device: PyTorch device string, e.g. "cpu" or "cuda:0".

    Returns:
        inputs:  Tensor of shape (batch_size, context_length)
        targets: Tensor of shape (batch_size, context_length)

    For each sampled start index i:
        inputs[row]  = x[i : i + context_length]
        targets[row] = x[i + 1 : i + 1 + context_length]
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")

    if len(x) < context_length + 1:
        raise ValueError(
            "x is too short: need at least context_length + 1 tokens "
            f"but got len(x)={len(x)} and context_length={context_length}"
        )

    # Valid start positions are [0, len(x) - context_length - 1]
    # randint upper bound is exclusive, so +1 gives:
    # [0, len(x) - context_length)
    starts = np.random.randint(
        0,
        len(x) - context_length,
        size=batch_size,
    )
    # slice x and then do list comprehension to construct the batch of input and target sequences
    # join a sequence of array into a new axis using np.stack
    inputs = np.stack(
        [x[i : i + context_length] for i in starts],
        axis=0,
    )
    targets = np.stack(
        [x[i + 1 : i + 1 + context_length] for i in starts],
        axis=0,
    )

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets