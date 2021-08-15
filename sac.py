import torch
from torch.nn.functional import mse_loss
import model
from torch.optim import Adam
import torch.nn.functional as F

gamma = 0.1
tau = 0.8

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


        next_action, log_pi, _ = self.actor.sample(state)

        #V 업데이트
        v_current = self.targetVnet.forward(state)

        q1,q2 = self.Qnet.forward(state,next_action)
        v_target = min(q1,q2) - log_pi
        v_loss = F.mse_loss(v_current,v_target) # v loss 계산하기

        self.VnetOptimizer.zero_grad()
        v_loss.backward()
        self.VnetOptimizer.step()

        #Q 업데이트

        # with torch.no_grad():
        #     next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
        #     qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        #     min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        #     next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_loss = qf1_loss + qf2_loss

        q_target = reward + gamma * self.targetVnet(state)

        q1_loss = F.mse_loss(q1,q_target)
        q2_loss = F.mse_loss(q2,q_target)


        self.QnetOptimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.QnetOptimizer.step()


        #actor 업데이트
            #log pi 와 Q를 구한다.
        # next_action, log_pi, _ = self.actor.sample(state)
        # q1,q2 = self.Qnet.forward(state,next_action)
        q = min(q1,q2)

        policyLoss = (log_pi - q) #.mean 도 한다는데 이건 batch로 할때 넣어야 할듯

        self.actorOptimizer.zero_grad()
        policyLoss.backward() # 이것도 우선 사용가능하게 나중에 위에꺼 고치기
        self.actorOptimizer.step()

        #target V 업데이트 -> 이거 매 스텝마다 하는건가?

        for target_param, param in zip(self.targetVnet.parameters(), self.Vnet.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        

    def sample(self, state) :
        
        action, _, _ = self.actor.sample(state) 
        q1,q2 = self.Qnet.forward(state, action)
        v = self.Vnet.forward(state)

        return action, q1, q2, v

    

    
    