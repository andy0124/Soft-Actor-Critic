import torch
from torch.nn.functional import mse_loss
import model
from torch.optim import Adam

class SAC(object):
    def __init__(self,stateNum, actionNum, hiddenNum, learningRate) -> None:
        super().__init__()
        
        #policy 모델 & optimizer 설정
        self.actor = model.StochasticPolicy(stateNum, hiddenNum, actionNum)
        self.actorOptimizer = Adam(self.actor.parameters, learningRate)

        #Q 함수 네트워크 모델 & optimizer 설정
        self.Qnet = model.DoubleQnetwork(actionNum, stateNum, hiddenNum)
        self.QnetOptimizer = Adam(self.Qnet.parameters, learningRate)

        #V 함수 네트워크 모델 & optimizer 설정
        self.Vnet = model.valueNetwork(stateNum, hiddenNum)
        self.VnetOptimizer = Adam(self.Vnet.parameters, learningRate)

        # target V 설정
        self.targetVnet = self.Vnet


    def updateParameter(self, transition) :

        state = transition[0]
        action = transition[1]
        reward = transition[2]
        next_state = transition[3]

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)


        #V 업데이트

        
        v_loss = 3 # v loss 계산하기

        self.VnetOptimizer.zero_grad()
        v_loss.backward()
        self.VnetOptimizer.step()

        #Q 업데이트




        q1_loss = 1
        q2_loss = 2 #q1,q2 둘다 loss 계산하기


        self.QnetOptimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.QnetOptimizer.step()


        #actor 업데이트
            #log pi 와 Q를 구한다.
        next_action, log_pi, _ = self.actor.sample(state)
        q1,q2 = self.Qnet.forward(state,next_action)
        q = min(q1,q2)

        policyLoss = (log_pi - q) #.mean 도 한다는데 이건 batch로 할때 넣어야 할듯

        self.actorOptimizer.zero_grad()
        policyLoss.backward() # 이것도 우선 사용가능하게 나중에 위에꺼 고치기
        self.actorOptimizer.step()
        
        




        #V 업데이트


        pass


    def updateTagetV(self) :
        #이렇게 하는게 맞는지 확인하기
        self.targetVnet = self.Vnet

    

    
    