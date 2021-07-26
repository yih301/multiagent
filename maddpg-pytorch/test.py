import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("env_id", help="Name of environment")
parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
parser.add_argument("--n_rollout_threads", default=1, type=int)
parser.add_argument("--n_training_threads", default=6, type=int)
parser.add_argument("--buffer_length", default=int(1e6), type=int)
parser.add_argument("--n_episodes", default=25000, type=int)
parser.add_argument("--episode_length", default=25, type=int)
parser.add_argument("--steps_per_update", default=100, type=int)
parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
parser.add_argument("--n_exploration_eps", default=25000, type=int)
parser.add_argument("--init_noise_scale", default=0.3, type=float)
parser.add_argument("--final_noise_scale", default=0.0, type=float)
parser.add_argument("--save_interval", default=1000, type=int)
parser.add_argument("--hidden_dim", default=64, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--tau", default=0.01, type=float)
parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
parser.add_argument("--discrete_action", default=True,
                        type=bool)

config = parser.parse_args()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
maddpg = MADDPG.init_from_save("models\\simple_adversary\\torchversion\\run7\\model.pt")
done = [False]
obs = env.reset()
print(obs)
while not done[0]:
    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
    print(torch_obs)
    #pdb.set_trace()
    # get actions as torch Variables
    torch_agent_actions = maddpg.step(torch_obs, explore=True)
    # convert actions to numpy arrays
    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
    # rearrange actions to be per environment
    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
    obs_n, reward_n, done, info_n = env.step(action)
    #print(obs_n)
    #print(done)
    env.render(mode='human')
pdb.set_trace()