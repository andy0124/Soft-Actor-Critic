import random
import numpy as np

class replayBuffer(object):
    def __init__(self,buffersize) -> None:
        super().__init__()
        self.buffer = list()
        self.buffersize = buffersize


    def push(self, state, action, reward, nextState):
        # 뭔가 제한 사항이 있는거 같은데
        if(len(self.buffer) >= self.buffersize) :
            del self.buffer[0]
        self.buffer.append((state,action.detach().numpy(),reward,nextState))

    def sample(self,samplesize):
        batch = random.sample(self.buffer, samplesize)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state
         