class replayBuffer(object):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = list()

    def push(self, state, action, reward, nextState):
        self.buffer.append((state,action,reward,nextState))

    def sample():
        pass