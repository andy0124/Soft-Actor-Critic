import gym
import replay
import sac
import torch

minimum_batch_size = 300

batch_size = 256

hiddenlayer_size = 256

learningRate = 0.8

if __name__ == '__main__':

    env = gym.make('HalfCheetah-v2')
    env.reset()

    rpb = replay.replayBuffer(1000000)
    sacModel = sac.SAC(env.observation_space.shape[0], env.action_space.shape[0],hiddenlayer_size,learningRate)

    for episode in range(100) :
        observation = env.reset()
        pastObs = observation
        step = 0
        while True :
            env.render()
            
            #action = env.action_space.sample() # Todo : 이부분은 나중에 sac로 대체할거다

            action = sacModel.sample(pastObs)
            observation, reward, done, info = env.step(action.detach().cpu().numpy()[0])
            if done:
                print("Episode finished after {} timesteps".format(step+1))
                break

            #replay에 transition 넣기
            rpb.push(pastObs,action, reward, observation)

            # 파리미터 학습
            #근데 원래 에피소드 중에 업데이트 하는건가?
            if len(rpb.buffer)> minimum_batch_size :
                state, action, reward, next_state = rpb.sample(batch_size)
                sacModel.updateParameter(state, action, reward, next_state)



            #할거 다하고 observation을 pastObs에 넣기
            pastObs = observation
            step = step + 1

        
        
        



    env.close()