import random

class replayBuffer(object):
    def __init__(self,buffersize) -> None:
        super().__init__()
        self.buffer = list()
        self.buffersize = buffersize


    def push(self, state, action, reward, nextState):
        # 뭔가 제한 사항이 있는거 같은데
        if(len(self.buffer) >= self.buffersize) :
            del self.buffer[0]
        self.buffer.append((state,action,reward,nextState))

    def sample(self,samplesize):
        return random.sample(self.buffer, samplesize)
         