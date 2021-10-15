import torch
import torch.utils.data as data_utils
import pickle
import numpy as np
import os
from PIL import Image
import pdb
import torchvision.models as pretrain
import torchvision.transforms as transforms

def rank_collate_func(items):
    item_list = [[], [], [], []]
    for item in items:
        item_list[0].append(item[0].float())
        item_list[2].append(item[2].float())
        item_list[1].append(item[1])
        item_list[3].append(item[3])
    item_list[1] = torch.Tensor(item_list[1]).float()
    item_list[3] = torch.Tensor(item_list[3]).float()
    return item_list

class RankingLimitDataset(data_utils.Dataset):
    def __init__(self, traj_files, pair_nums, state_dim, action_dim, mode='state_only', traj_len=25, seed=1234, agent_num=3):
        self.pairs1 = []
        self.pairs2 = []
        self.trajs = []
        self.jump_steps = 1

        np.random.seed(seed)

        self.traj_index = 0
        for i in range(len(traj_files)):
            loaded_data = pickle.load(open(traj_files[i], 'rb'))
            self.trajs += loaded_data['traj']
            all_pairs = self.get_all_pairs(loaded_data['traj'], loaded_data['reward'], traj_len)
            if pair_nums is not None:
                sample_idx1 = np.random.choice(len(all_pairs), pair_nums[i], replace=False)
                self.pairs1 += [all_pairs[idx] for idx in sample_idx1]
                sample_idx2 = np.random.choice(len(all_pairs), pair_nums[i], replace=False)
                self.pairs2 += [all_pairs[idx] for idx in sample_idx2]
            else:
                self.pairs1 += all_pairs
                self.pairs2 += all_pairs
        self.traj_len = traj_len
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        ## different for different envs
        self.action_limit = [-1., 1.]
        np.random.seed(seed)
        self.pairs1 = np.random.permutation(self.pairs1)
        self.pairs2 = np.random.permutation(self.pairs2)
        

    def get_all_pairs(self, trajs, rewards, traj_len):
        all_pairs = []
        for i in range(len(trajs)):
            if traj_len > 0:
                for j in range(max(1, len(rewards[i])-traj_len*self.jump_steps+1)):
                    all_pairs.append([self.traj_index,j,np.sum(rewards[i][j:j+traj_len*self.jump_steps:self.jump_steps],axis=0)])
                    #all_pairs.append([self.traj_index,j,rewards[i][j:j+traj_len*self.jump_steps:self.jump_steps][0]])  #changed this
            else:
                all_pairs.append([self.traj_index,0,np.sum(rewards[i][::self.jump_steps],axis=0)])
                #all_pairs.append([self.traj_index,0,rewards[i][::self.jump_steps]])  #changed this
            self.traj_index += 1
        return all_pairs

    def __getitem__(self, index):
        traj1 = self.trajs[int(self.pairs1[index][0])]
        traj2 = self.trajs[int(self.pairs2[index][0])]
        rew1 = self.pairs1[index][2]
        rew2 = self.pairs2[index][2]
        ret_traj1 = []
        ret_traj2 = []
        for i in range(0, min(self.traj_len*self.jump_steps, len(traj1)), self.jump_steps):
            if self.mode == 'state_only':
                ret_traj1.append(traj1[int(self.pairs1[index][1])+i][0:self.agent_num][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj1.append(np.concatenate([traj1[int(self.pairs1[index][1])+i][0:self.agent_num][0:self.state_dim], traj1[int(self.pairs1[index][1])+i+1][0:self.agent_num][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                r_pairs = np.array(traj1[int(self.pairs1[index][1])+i])
                r_pairs[self.state_dim:] = np.clip(r_pairs[self.state_dim:], self.action_limit[0], self.action_limit[1])
                ret_traj1.append(r_pairs)
            else:
                raise NotImplementedError
        ret_traj1 = np.array(ret_traj1)

        for i in range(0, min(self.traj_len*self.jump_steps, len(traj2)), self.jump_steps):
            if self.mode == 'state_only':
                ret_traj2.append(traj2[int(self.pairs2[index][1])+i][0:self.agent_num][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj2.append(np.concatenate([traj2[int(self.pairs2[index][1])+i][0:self.agent_num][0:self.state_dim], traj2[int(self.pairs2[index][1])+i+1][0:self.agent_num][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                r_pairs = np.array(traj2[int(self.pairs2[index][1])+i])
                r_pairs[self.state_dim:] = np.clip(r_pairs[self.state_dim:], self.action_limit[0], self.action_limit[1])
                ret_traj2.append(r_pairs)
            else:
                raise NotImplementedError
        ret_traj2 = np.array(ret_traj2)

        return torch.from_numpy(ret_traj1), rew1, torch.from_numpy(ret_traj2), rew2

    def __len__(self):
        return len(self.pairs1)

class RankingLimitTrajDataset(RankingLimitDataset):
    def __init__(self, traj_files, pair_nums, state_dim, action_dim, mode='state_only', seed=1234):
        super(RankingLimitTrajDataset, self).__init__(traj_files, pair_nums, state_dim, action_dim, mode, -1, seed)

    def __getitem__(self, index):
        traj1 = self.trajs[int(self.pairs1[index][0])]
        traj2 = self.trajs[int(self.pairs2[index][0])]
        rew1 = self.pairs1[index][2]
        rew2 = self.pairs2[index][2]
        ret_traj1 = []
        ret_traj2 = []
        for i in range(0, len(traj1)-1, self.jump_steps):
            if self.mode == 'state_only':
                ret_traj1.append(traj1[i][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj1.append(np.concatenate([traj1[i][0:self.agent_num][0:self.state_dim], traj1[i+1][0:self.agent_num][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                ret_traj1.append(traj1[i])
            else:
                raise NotImplementedError
        ret_traj1 = np.array(ret_traj1)

        for i in range(0, len(traj2)-1, self.jump_steps):
            if self.mode == 'state_only':
                ret_traj2.append(traj2[i][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj2.append(np.concatenate([traj2[i][0:self.agent_num][0:self.state_dim], traj2[i+1][0:self.agent_num][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                ret_traj2.append(traj2[i])
            else:
                raise NotImplementedError
        ret_traj2 = np.array(ret_traj2)

        return torch.from_numpy(ret_traj1), rew1, torch.from_numpy(ret_traj2), rew2

'''
def rank_collate_func(items):
    item_list = [[], [], [], []]
    for item in items:
        item_list[0].append(item[0].float())
        item_list[2].append(item[2].float())
        item_list[1].append(item[1])
        item_list[3].append(item[3])
    item_list[1] = torch.Tensor(item_list[1]).float()
    item_list[3] = torch.Tensor(item_list[3]).float()
    return item_list
class RankingLimitDataset(data_utils.Dataset):
    def __init__(self, traj_files, pair_nums, state_dim, action_dim, mode='state_only', traj_len=1, seed=1234):
        self.pairs1 = []
        self.pairs2 = []
        self.trajs = []
        self.jump_steps = 1
        np.random.seed(seed)
        for i in range(len(traj_files)):
            loaded_data = pickle.load(open(traj_files[i], 'rb'))
            self.trajs += loaded_data['traj']
            all_pairs = self.get_all_pairs(loaded_data['traj'], loaded_data['reward'], traj_len)
            pdb.set_trace()
            if pair_nums is not None:
                sample_idx1 = np.random.choice(len(all_pairs), pair_nums[i], replace=False)
                self.pairs1 += [all_pairs[idx] for idx in sample_idx1]
                sample_idx2 = np.random.choice(len(all_pairs), pair_nums[i], replace=False)
                self.pairs2 += [all_pairs[idx] for idx in sample_idx2]
            else:
                self.pairs1 += all_pairs
                self.pairs2 += all_pairs
        self.traj_len = traj_len
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        np.random.seed(seed)
        self.pairs1 = np.random.permutation(self.pairs1)
        self.pairs2 = np.random.permutation(self.pairs2)
    def get_all_pairs(self, trajs, rewards, traj_len):
        all_pairs = []
        for i in range(len(trajs)):
            if traj_len > 0:
                for j in range(len(rewards[i])-traj_len*self.jump_steps+1):
                    all_pairs.append([i,j,np.sum(rewards[i][j:j+traj_len*self.jump_steps:self.jump_steps])])
                    pdb.set_trace()
            else:
                all_pairs.append([i,0,np.sum(rewards[i][::self.jump_steps])])
        return all_pairs
    def __getitem__(self, index):
        traj1 = self.trajs[int(self.pairs1[index][0])]
        traj2 = self.trajs[int(self.pairs2[index][0])]
        rew1 = self.pairs1[index][2]
        rew2 = self.pairs2[index][2]
        ret_traj1 = []
        ret_traj2 = []
        for i in range(0, self.traj_len*self.jump_steps, self.jump_steps):
            if self.mode == 'state_only':
                ret_traj1.append(traj1[int(self.pairs1[index][1])+i][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj1.append(np.concatenate([traj1[int(self.pairs1[index][1])+i][0:self.state_dim], traj1[int(self.pairs1[index][1])+i+1][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                ret_traj1.append(traj1[int(self.pairs1[index][1])+i])
            else:
                raise NotImplementedError
        ret_traj1 = np.array(ret_traj1)
        for i in range(0, self.traj_len*self.jump_steps, self.jump_steps):
            if self.mode == 'state_only':
                ret_traj2.append(traj2[int(self.pairs2[index][1])+i][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj2.append(np.concatenate([traj2[int(self.pairs2[index][1])+i][0:self.state_dim], traj2[int(self.pairs2[index][1])+i+1][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                ret_traj2.append(traj2[int(self.pairs2[index][1])+i])
            else:
                raise NotImplementedError
        ret_traj2 = np.array(ret_traj2)
        return torch.from_numpy(ret_traj1), rew1, torch.from_numpy(ret_traj2), rew2
    def __len__(self):
        return len(self.pairs1)
class RankingLimitTrajDataset(RankingLimitDataset):
    def __init__(self, traj_files, pair_nums, state_dim, action_dim, mode='state_only', seed=1234):
        super(RankingLimitTrajDataset, self).__init__(traj_files, pair_nums, state_dim, action_dim, mode, -1, seed)
    def __getitem__(self, index):
        traj1 = self.trajs[int(self.pairs1[index][0])]
        traj2 = self.trajs[int(self.pairs2[index][0])]
        rew1 = self.pairs1[index][2]
        rew2 = self.pairs2[index][2]
        ret_traj1 = []
        ret_traj2 = []
        for i in range(0, len(traj1)-1, self.jump_steps):
            if self.mode == 'state_only':
                ret_traj1.append(traj1[i][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj1.append(np.concatenate([traj1[i][0:self.state_dim], traj1[i+1][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                ret_traj1.append(traj1[i])
            else:
                raise NotImplementedError
        ret_traj1 = np.array(ret_traj1)
        for i in range(0, len(traj2)-1, self.jump_steps):
            if self.mode == 'state_only':
                ret_traj2.append(traj2[i][0:self.state_dim])
            elif self.mode == 'state_pair':
                ret_traj2.append(np.concatenate([traj2[i][0:self.state_dim], traj2[i+1][0:self.state_dim]], axis=0))
            elif self.mode == 'state_action':
                ret_traj2.append(traj2[i])
            else:
                raise NotImplementedError
        ret_traj2 = np.array(ret_traj2)
        return torch.from_numpy(ret_traj1), rew1, torch.from_numpy(ret_traj2), rew2
'''