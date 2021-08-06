import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
    
epsilon = 1e-6   
    
class StochasticPolicy(nn.Module) : #Gausian
    def __init__(self, stateInputNum, HiddenLayerNum, actionInputNum) -> None:
        super(StochasticPolicy,self).__init__()
        self.hidden1 = nn.Linear(stateInputNum, HiddenLayerNum)
        self.hidden2 = nn.Linear(HiddenLayerNum, HiddenLayerNum)

        self.meanlayer = nn.Linear(HiddenLayerNum, actionInputNum)
        self.stdlayer = nn.Linear(HiddenLayerNum, actionInputNum)
    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        mean = self.meanlayer(x)
        std = self.stdlayer(x)

        return mean, std

    def sample(self, state):
        mean , std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        #여기서 부터 좀...
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



class DoubleQnetwork(nn.Module) :
    def __init__(self, actionInputNum, stateInputNum, HiddenlayerNum) -> None:
        super().__init__()
        #first Q
        self.hiddenlayer1 = nn.Linear(stateInputNum + actionInputNum, HiddenlayerNum)
        self.hiddenlayer2 = nn.Linear(HiddenlayerNum, HiddenlayerNum)
        self.hiddenlayer3 = nn.Linear(HiddenlayerNum, 1)
        #second Q
        self.hiddenlayer4 = nn.Linear(stateInputNum + actionInputNum, HiddenlayerNum)
        self.hiddenlayer5 = nn.Linear(HiddenlayerNum, HiddenlayerNum)
        self.hiddenlayer6 = nn.Linear(HiddenlayerNum, 1)

    def forward(self, state, action):
        x = F.relu(self.hiddenlayer1(torch.cat(state,action)))
        x = F.relu(self.hiddenlayer2(x))
        x = self.hiddenlayer3(x)

        y = F.relu(self.hiddenlayer4(torch.cat(state,action)))
        y = F.relu(self.hiddenlayer5(y))
        y = self.hiddenlayer6(y)

        return x , y

    
class valueNetwork(nn.Module) :
    def __init__(self, stateInputNum, HiddenLayerNum,):
        super().__init__()
        self.hidden1 = nn.Linear(stateInputNum, HiddenLayerNum)
        self.hidden2 = nn.Linear(HiddenLayerNum, HiddenLayerNum)
        self.hidden3 = nn.Linear(HiddenLayerNum, 1)

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x)

        return x

        



