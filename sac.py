import torch
from torch.nn.functional import mse_loss
import model
from torch.optim import Adam
import torch.nn.functional as F

gamma = 0.1
tau = 1.0

class SAC(object):
    def __init__(self,stateNum, actionNum, hiddenNum, learningRate) -> None:
        super().__init__()
        
        #policy 모델 & optimizer 설정
        self.actor = model.StochasticPolicy(stateNum, hiddenNum, actionNum)
        self.actorOptimizer = Adam(self.actor.parameters(), learningRate)

        #Q 함수 네트워크 모델 & optimizer 설정
        self.Qnet1 = model.DoubleQnetwork(actionNum, stateNum, hiddenNum)
        self.Qnet1Optimizer = Adam(self.Qnet1.parameters(), learningRate)

        self.Qnet2 = model.DoubleQnetwork(actionNum, stateNum, hiddenNum)
        self.Qnet2Optimizer = Adam(self.Qnet2.parameters(), learningRate)

        #V 함수 네트워크 모델 & optimizer 설정
        self.Vnet = model.valueNetwork(stateNum, hiddenNum)
        self.VnetOptimizer = Adam(self.Vnet.parameters(), learningRate)

        # target V 설정
        self.targetVnet = model.valueNetwork(stateNum, hiddenNum)
        for target_param, param in zip(self.targetVnet.parameters(), self.Vnet.parameters()):
            target_param.data.copy_(param.data)


    def updateParameter(self, state, action, reward, next_state, batch_size):

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)


        reward = torch.reshape(reward, (batch_size, 1))

        next_action, log_pi, _ = self.actor.sample(state)

        #V 업데이트
        v_current = self.targetVnet.forward(state)

        # q1,q2 = self.Qnet.forward(state,next_action)
        q1 = self.Qnet1.forward(state,next_action)
        q2 = self.Qnet2.forward(state,next_action)
        q = torch.min(q1,q2)
        v_target = q - log_pi
        v_loss = F.mse_loss(v_current,v_target) # v loss 계산하기

        self.VnetOptimizer.zero_grad()
        v_loss.backward(retain_graph=True)
        self.VnetOptimizer.step()

        #Q 업데이트
        targetV = self.targetVnet.forward(next_state)
        q_target = reward + gamma * targetV

        q1 = self.Qnet1.forward(state,action)
        q2 = self.Qnet2.forward(state,action)



        q1_loss = F.mse_loss(q1,q_target)
        q2_loss = F.mse_loss(q2,q_target)


        self.Qnet1Optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.Qnet1Optimizer.step()

        self.Qnet2Optimizer.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.Qnet2Optimizer.step()



        #actor 업데이트

        policyLoss = (0.2*log_pi - q).mean()

        self.actorOptimizer.zero_grad()
        policyLoss.backward()
        self.actorOptimizer.step()

        #target V 업데이트
        for target_param, param in zip(self.targetVnet.parameters(), self.Vnet.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        

    def sample(self, state) :
        state = torch.FloatTensor(state)
        action, _, _ = self.actor.sample(state) 
        

        return action

    

    
    