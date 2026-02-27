import math

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Compute learning rate at step t using:
    1) linear warmup
    2) cosine annealing
    3) post-annealing constant min lr

    Args:
        t: current training step
        alpha_max: maximum learning rate
        alpha_min: minimum learning rate
        T_w: number of warmup steps
        T_c: number of cosine annealing steps (inclusive upper bound)

    Returns:
        Learning rate at step t.
    """
    if T_w < 0 or T_c < 0:
        raise ValueError("T_w and T_c must be non-negative.")
    if T_w > T_c:
        raise ValueError("T_w must be <= T_c.")
    if alpha_max < 0 or alpha_min < 0:
        raise ValueError("Learning rates must be non-negative.")

    # Warm-up: if t < T_w, alpha_t = (t / T_w) * alpha_max
    if t < T_w:
        # If T_w == 0, warmup region is empty, but this branch won't be entered.
        return (t / T_w) * alpha_max

    # Cosine annealing: if T_w <= t <= T_c
    if t <= T_c:
        # Special case: if T_w == T_c, cosine interval collapses to one point.
        if T_c == T_w:
            return alpha_min

        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)

    # Post-annealing: if t > T_c
    return alpha_min