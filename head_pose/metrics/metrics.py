import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


class Metric:
    def __init__(self, n_classes=2):

        self.tp = np.array([0] * n_classes)
        self.fp = np.array([0] * n_classes)
        self.fn = np.array([0] * n_classes)

        self.n_classes = n_classes

    def calc_running_score(self, labels, outputs):

        # TODO
        labels = np.eye(self.n_classes)[labels]
        outputs = np.eye(self.n_classes)[outputs]

        tp = np.sum(labels * outputs, axis=0)
        fp = np.sum(outputs, axis=0) - tp
        fn = np.sum(labels, axis=0) - tp

        self.tp = self.tp + tp
        self.fp = self.fp + fp
        self.fn = self.fn + fn

    def compute(self):

        # f1 macro
        f1 = []
        for i in range(self.n_classes):
            f1.append(self.tp[i] / (self.tp[i] + 0.5 * (self.fp[i] + self.fn[i]) + 1e-3))

        self.tp = np.array([0] * self.n_classes)
        self.fp = np.array([0] * self.n_classes)
        self.fn = np.array([0] * self.n_classes)

        return np.mean(f1)

    def calc_running_score_samplewise(self, labels, outputs):

        mae = np.mean(np.abs(labels - outputs), axis=1)

        return mae.tolist()

    def reset(self):
        self.tp = np.array([0] * self.n_classes)
        self.fp = np.array([0] * self.n_classes)
        self.fn = np.array([0] * self.n_classes)
        return True
