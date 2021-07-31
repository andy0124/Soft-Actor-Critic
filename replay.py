class replayBuffer(object):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = list()

    def push(self, sample):
        self.buffer.append(sample)

    def sample():
        pass