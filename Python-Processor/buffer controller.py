import numpy as np
from collections import deque, Counter


class SSVEPInputBuffer:
    """
    Turns streaming CCA outputs into stable input events.
    """

    def __init__(
        self,
        targets=[7, 11, 13, 17]
        history_size=10,
        min_confidence=0.5,
        min_consensus=0.6,
        cooldown_steps=8,
    ):
        self.targets = targets
        self.history = deque(maxlen=history_size)

        self.min_confidence = min_confidence
        self.min_consensus = min_consensus

        self.cooldown_steps = cooldown_steps
        self.cooldown = 0

        self.last_output = None

    def update(self, cca_scores):
        """
        cca_scores: array of correlation values per target
        """

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        cca_scores = np.asarray(cca_scores)

        idx = int(np.argmax(cca_scores))
        pred = self.targets[idx]
        confidence = cca_scores[idx]

        self.history.append(pred)

        counts = Counter(self.history)
        winner, votes = counts.most_common(1)[0]

        consensus = votes / len(self.history)

        # decision rule
        if (
            confidence >= self.min_confidence
            and consensus >= self.min_consensus
        ):
            if winner != self.last_output:
                self.last_output = winner
                self.history.clear()
                self.cooldown = self.cooldown_steps
                return winner

        return None
