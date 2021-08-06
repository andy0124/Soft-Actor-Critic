import torch
from torch.nn.functional import mse_loss
import model
from torch.optim import Adam

class SAC(object):
    def __init__(self,stateNum, actionNum, hiddenNum, learningRate) -> None:
        super().__init__()
        
        self.actor = model.StochasticPolicy(stateNum, hiddenNum, actionNum)
        self.actorOptimizer = Adam(self.actor.parameters, learningRate)

        self.Qnet = model.DoubleQnetwork(actionNum, stateNum, hiddenNum)
        self.QnetOptimizer = Adam(self.Qnet.parameters, learningRate)

        self.Vnet = model.valueNetwork(stateNum, hiddenNum)
        self.VnetOptimizer = Adam(self.Vnet.parameters, learningRate)

        self.targetVnet = self.Vnet

        

        
    def _criticLoss(self,state):
        mse_loss()
        pass


    def updateParameter(self, transition) :
        
        
        
        pass
        

    
    