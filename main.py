import numpy as np
import gym
import model
import replay


env = gym.make('HalfCheetah-v2')
env.reset()

for episode in range(100) :
    observation = env.reset()
    for step in range(100) :
        env.render()

        action = env.action_space.sample() # Todo : 이부분은 나중에 sac로 대체할거다
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break



env.close()