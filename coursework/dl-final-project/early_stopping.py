"""Early stopping (plan Section 6/7): stop training when the monitored metric hasn't
improved for `patience` consecutive checks -- keeps mini-tier training short
(PDF Section 12.1's computational-cost constraint) without hand-picking an
epoch count in advance.
"""


class EarlyStopper:
    """Tracks the best value seen so far and how many checks have passed without improvement."""

    def __init__(self, patience: int, mode: str = "min"):
        """Configure `patience` (checks allowed without improvement) and `mode` ('min'/'max')."""
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.is_best = False

    def step(self, value: float) -> bool:
        """Record `value`; return True if training should stop now."""
        improved = self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        )
        self.is_best = improved
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
