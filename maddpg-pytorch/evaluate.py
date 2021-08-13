import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import pdb
import os
import pickle


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    savefile1 = {}
    savefile2 = {}
    savefile3 = {}
    allfile = {}
    fileobs = [[],[],[]]
    filerewards = [[],[],[]]
    filestate=[[],[],[]]
    fileallo = []
    fileallr = []
    filealls = []
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        saveobs = [[],[],[]]
        saverewards = [[],[],[]]
        savestate=[[],[],[]]
        saveallo = []
        saveallr = []
        savealls = []
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            saveobs[0].append(np.array(obs[0]))
            saveobs[1].append(np.array(obs[1]))
            saveobs[2].append(np.array(obs[2]))
            saveallo.append(np.array(obs))
            saverewards[0].append(np.array(rewards[0]))
            saverewards[1].append(np.array(rewards[1]))
            saverewards[2].append(np.array(rewards[2]))
            saveallr.append(np.array(np.array(rewards)))
            statearray1 = [env.world.agents[0].state.p_pos[0], env.world.agents[0].state.p_pos[1], 
                            env.world.agents[0].state.p_vel[0], env.world.agents[0].state.p_vel[1],
                            env.world.agents[0].state.c[0], env.world.agents[0].state.c[1]]
            statearray2 = [env.world.agents[1].state.p_pos[0], env.world.agents[1].state.p_pos[1], 
                            env.world.agents[1].state.p_vel[0], env.world.agents[1].state.p_vel[1],
                            env.world.agents[1].state.c[0], env.world.agents[1].state.c[1]]
            statearray3 = [env.world.agents[2].state.p_pos[0], env.world.agents[2].state.p_pos[1], 
                            env.world.agents[2].state.p_vel[0], env.world.agents[2].state.p_vel[1],
                            env.world.agents[2].state.c[0], env.world.agents[2].state.c[1]]
            savestate[0].append(np.array(statearray1))
            savestate[1].append(np.array(statearray2))
            savestate[2].append(np.array(statearray3))
            savealls.append(np.array([np.array(statearray1),np.array(statearray2),np.array(statearray3)]))
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        fileobs[0].append(saveobs[0])
        fileobs[1].append(saveobs[1])
        fileobs[2].append(saveobs[2])
        fileallo.append(saveallo)
        filerewards[0].append(saverewards[0])
        filerewards[1].append(saverewards[1])
        filerewards[2].append(saverewards[2])
        fileallr.append(saveallr)
        filestate[0].append(savestate[0])
        filestate[1].append(savestate[1])
        filestate[2].append(savestate[2])
        filealls.append(savealls)
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
    savefile1['obs'] =fileobs[0]
    savefile1['reward'] = filerewards[0]
    savefile1['traj'] = filestate[0]
    savefile2['obs'] =fileobs[1]
    savefile2['reward'] = filerewards[1]
    savefile2['traj'] = filestate[1]
    savefile3['obs'] =fileobs[2]
    savefile3['reward'] = filerewards[2]
    savefile3['traj'] = filestate[2]
    allfile['obs'] =fileallo
    allfile['reward'] = fileallr
    allfile['traj'] = filealls
    file_path = (Path('./models') / config.env_id / config.model_name /
                ('run%i' % config.run_num))
    with open(os.path.join(file_path,'1.pkl'), "wb") as f:
        pickle.dump(savefile1, f)
    with open(os.path.join(file_path,'2.pkl'), "wb") as f:
        pickle.dump(savefile1, f)
    with open(os.path.join(file_path,'3.pkl'), "wb") as f:
        pickle.dump(savefile1, f)
    with open(os.path.join(file_path,'all.pkl'), "wb") as f:
        pickle.dump(allfile, f)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", default=False,
                        type=bool,
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()
    #pdb.set_trace()

    run(config)