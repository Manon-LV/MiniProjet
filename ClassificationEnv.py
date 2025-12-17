import numpy as np

class ClassificationEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_classes = len(np.unique(y))
        self.current = 0

    def reset(self):
        self.current = 0
        return self.X[self.current]

    def step(self, action):
        correct = int(action == self.y[self.current])
        reward = 1 if correct else -1
        done = self.current == self.n_samples - 1
        obs = self.X[self.current + 1] if not done else None
        self.current += 1
        return obs, reward, done, {"label": self.y[self.current-1], "pred": action}
