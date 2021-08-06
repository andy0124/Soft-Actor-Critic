import numpy as np
import gym
import model
import replay
import sac


if __name__ == '__main__':

    env = gym.make('HalfCheetah-v2')
    env.reset()

    rpb = replay.replayBuffer()
    sacModel = sac.SAC(1,1,1,1)

    for episode in range(100) :
        observation = env.reset()
        pastObs = observation
        step = 0
        while True :
            env.render()
            
            action = env.action_space.sample() # Todo : 이부분은 나중에 sac로 대체할거다
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step+1))
                break

            #replay에 transition 넣기
            rpb.push((pastObs,action, reward, observation))





            #할거 다하고 observation을 pastObs에 넣기
            pastObs = observation
            step = step + 1

        
        # 파리미터 학습
        



    env.close()