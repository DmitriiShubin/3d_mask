import numpy as np


class PostProcessing:
    def __init__(self, threshold=0.5):

        self.threshold = threshold

    def run(self, pred: np.array):

        pred = np.round(pred, 2)

        pred = np.argmax(pred, axis=1)

        return pred
