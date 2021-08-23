import gym
import replay
import sac
import torch
from numpy import round
import numpy as np

minimum_batch_size = 300

batch_size = 256

hiddenlayer_size = 256

learningRate = 0.8

if __name__ == '__main__':

    env = gym.make('HalfCheetah-v2')
    env.reset()

    rpb = replay.replayBuffer(1000000)
    sacModel = sac.SAC(env.observation_space.shape[0], env.action_space.shape[0],hiddenlayer_size,learningRate)

    # Training Loop
    total_numsteps = 0
    updates = 0

    

    for episode in range(100) :
        episode_reward = 0
        episode_steps = 0
        
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
                sacModel.updateParameter(state, action, reward, next_state, batch_size)



            #할거 다하고 observation을 pastObs에 넣기
            pastObs = observation
            step += 1

            total_numsteps += 1
            episode_reward += reward

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(episode, total_numsteps, step, round(np.sum(episode_reward), 2)))
        
        if episode % 10 == 0 :
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:

                    action = sacModel.sample(state)
                    next_state, reward, done, _ = env.step(action.detach().cpu().numpy()[0])
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")        
        



    env.close()