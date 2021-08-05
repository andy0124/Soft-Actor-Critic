import torch
from torch.nn.functional import mse_loss
import model
import replay
from torch.optim import Adam

class sac(object):
    def __init__(self,stateNum, actionNum, hiddenNum, learningRate) -> None:
        super().__init__()
        
        self.actor = model.StochasticPolicy(stateNum, hiddenNum, actionNum)
        self.actorOptimizer = Adam(self.actor.parameters, learningRate)

        self.critic = model.DoubleQnetwork(actionNum, stateNum, hiddenNum)
        self.criticOptimizer = Adam(self.critic.parameters, learningRate)

        self.targetCritic = model.DoubleQnetwork(actionNum, stateNum, hiddenNum)

        

        
    def _criticLoss(self,state):
        mse_loss()
        

    
    