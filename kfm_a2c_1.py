import gym
import numpy as np
import random

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from tensorflow.keras.callbacks import TensorBoard
import os


def train_A2C(a2c_path, env):

    #env = make_atari_env('KungFuMaster-v0', n_envs=4, seed=0)

    #env = VecFrameStack(env, n_stack=4)
    # env = make_vec_env("KungFuMaster-v0", n_envs=4)

    log_path = os.path.join('Training', 'Logs')

    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    log_callback=TensorBoard("./dqn_logs")

    model.learn(total_timesteps=200000, callback=[log_callback])

    # a2c_path = os.path.join('Training', 'Saved_Models', '1000k_model')

    model.save(a2c_path)



def test_model():
    steps = 4
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()

        score = 0
        for step in range(1, steps+1):
            done = False

            while not done:
                # env.render()
                action, _state = model.predict(state)
                n_state, reward, done, info = env.step(action)
                state = n_state
                score += reward[0]*200
        print('Episode:{} Score:{}'.format(episode, score))


if __name__ == "__main__":
    #env = make_atari_env('KungFuMaster-v0', n_envs=1, seed=0)
    # env_record = gym.wrappers.Monitor(env, 'recording_a2c', force=True)
    #env = VecFrameStack(env, n_stack=4)
    # env = gym.make('KungFuMaster-v0')
    env = make_vec_env("KungFuMaster-v0", n_envs=4)
    
    a2c_path = os.path.join('Training', 'Saved_Models', '200k_model_3')

    train_A2C(a2c_path, env)

    # model = A2C.load(a2c_path, env)

    # steps = 4
    # episodes = 5
    # for episode in range(1, episodes+1):
    #     state = env.reset()

    #     score = 0
    #     for step in range(1, steps+1):
    #         done = False

    #         while not done:
    #             # env.render()
    #             action, _state = model.predict(state)
    #             n_state, reward, done, info = env.step(action)
    #             state = n_state
    #             score += reward[0]
    #     print('Episode:{} Score:{}'.format(episode, score))

   # test_model()
    env.close()

    #evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=False)

    #print(evaluation)

