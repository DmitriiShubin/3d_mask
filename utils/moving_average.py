class MovingAverage:
    def __init__(self, length: int):

        self.length = length  # length of the input window
        self.window = [0] * length
        self.average = 0

    def run(self, value: float) -> float:

        self.average -= self.window[0] / self.length
        self.window.pop(0)
        self.window.append(value)
        self.average += value / self.length

        return self.average
