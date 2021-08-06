class replayBuffer(object):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = list()

    def push(self, state, action, reward, nextState):
        # 뭔가 제한 사항이 있는거 같은데
        self.buffer.append((state,action,reward,nextState))

    def sample():
        pass