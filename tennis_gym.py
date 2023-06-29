import gym
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

#create tennis environment
env = gym.make('KungFuMaster-v0')
# env = gym.wrappers.Monitor(env, 'recording_3', video_callable=lambda episode_id: True, force=True)

# env = gym.wrappers.Monitor(env, 'recording', force=True)

height, width, channels = env.observation_space.shape
actions = env.action_space.n
# action_meanings = env.unwrapped.get_action_meanings()
# print('action meanings:', meanings)


# build keras deep learning model
def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(height, width, channels, actions)

print(model.summary())


# build keras-rl agent
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=.1, value_test=.2, nb_steps=200000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg', nb_actions=actions, nb_steps_warmup=20000)
    return dqn

dqn = build_agent(model, actions)
# learning rate: 0.001
dqn.compile(Adam(lr=1e-4))

# log_callback=TensorBoard("./dqn_logs")

# dqn.fit(env, nb_steps=200000, callbacks=[log_callback], visualize=False, verbose=2)

# dqn.save_weights('SavedWeights/200k_2/dqn_weights.h5f')

# load and test
dqn.load_weights('SavedWeights/200k/dqn_weights.h5f')

evaluations = dqn.test(env, nb_episodes=200, visualize=True)
plt.plot(evaluations.history['episode_reward'])
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('dqn_test_1.jpg')
print(np.mean(evaluations.history['episode_reward']))


# scores = dqn.test(env, nb_episodes=10, visualize=False)
# print(np.mean(scores.history['episode_reward']))

# dqn.save_weights('SavedWeights/50k/dqn_weights.h5f')

