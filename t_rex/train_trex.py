
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import gym

import models
import dataset

import pdb
import time
import imageio
import sys
#sys.path.append(r"C:\\Users\\Yilun\\Desktop\\Robot\\multi-agent\\maddpg-pytorch")  #added this for evaluate
sys.path.append(r"/iliad/u/yilunhao/multiagent/maddpg-pytorch")  #added this for evaluate
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import os
import pickle

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--train_demo_files', nargs='+', help='the path to the training demonstrations files')
parser.add_argument('--test_demo_files', nargs='+', help='the path to the testing demonstrations files')
parser.add_argument('--train_traj_nums', nargs='+', type=int, help='the number of trajectories for each training demonstration file')
parser.add_argument('--num_epochs', type=int, default=100, help='the path to the testing demonstrations files')
parser.add_argument('--mode', default='state_only', help='the mode of the reward function')
parser.add_argument('--dataset_mode', default='partial', help='the dataset mode')
parser.add_argument('--network_mode', default='single', help='the network mode')
parser.add_argument('--num_agent', type=int, default=3, help='the # of agent')
parser.add_argument('--trajdimension', type=int, default=6, help='the traj dimension')
parser.add_argument('--actiondimension', type=int, default=2, help='the action dimension')
parser.add_argument('--output_model_path', help='the output path for models and logs')
parser.add_argument('--agent', type=int, default=1, help='the decreased agent')
parser.add_argument('--iter', type=int, default=1, help='the running iteration')
args = parser.parse_args()

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

env = make_env(args.env_name, True)
test_env = make_env(args.env_name, True)

num_inputs = args.trajdimension  #env.observation_space.shape[0]
num_actions = args.actiondimension #s2 #env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset_mode == 'partial':
    train_dataset = dataset.RankingLimitDataset(args.train_demo_files, args.train_traj_nums, num_inputs, num_actions, mode=args.mode)
    test_dataset = dataset.RankingLimitDataset(args.test_demo_files, None, num_inputs, num_actions, mode=args.mode)
elif args.dataset_mode == 'traj':
    train_dataset = dataset.RankingTrajDataset(args.train_demo_files, args.train_traj_nums, num_inputs, num_actions, mode=args.mode)
    test_dataset = dataset.RankingTrajDataset(args.test_demo_files, None, num_inputs, num_actions, mode=args.mode)
else:
    raise NotImplementedError
print(len(train_dataset))
train_loader = data_utils.DataLoader(train_dataset, collate_fn=dataset.rank_collate_func, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data_utils.DataLoader(test_dataset, collate_fn=dataset.rank_collate_func, batch_size=1, shuffle=False, num_workers=4)

if args.network_mode == 'single':
    if args.mode == 'state_only':
        reward_net = models.singleRewardNet(num_inputs).float()
    elif args.mode == 'state_pair':
        reward_net = models.singleRewardNet(num_inputs*2).float()
    elif args.mode == 'state_action':
        reward_net = models.singleRewardNet(num_inputs+num_actions).float()
    else:
        raise NotImplementedError
else: 
    if args.mode == 'state_only':
        reward_net = models.ShareRewardNet(num_inputs).float()
    elif args.mode == 'state_pair':
        reward_net = models.ShareRewardNet(num_inputs*2).float()
    elif args.mode == 'state_action':
        reward_net = models.ShareRewardNet(num_inputs+num_actions).float()
    else:
        raise NotImplementedError
if use_gpu:
    reward_net = reward_net.cuda()
optimizer = optim.Adam(reward_net.parameters(), lr=0.001, weight_decay=0.0005)

best_acc = 0
for epoch in range(args.num_epochs):
    counter = [0]*args.num_agent
    acc_counter = [0]*args.num_agent
    if epoch % args.save_interval == 0:
        for iter_, data in enumerate(test_loader):
            traj1, rew1, traj2, rew2 = data
            if use_gpu:
                traj1, rew1, traj2, rew2 = [item.cuda() for item in traj1], rew1.cuda(), [item.cuda() for item in traj2], rew2.cuda()
            bs1 = len(traj1)
            bs2 = len(traj2)
            assert bs1 == bs2

            if args.network_mode == 'nnnsingle':
                pred_rew1 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj1], dim=0)
                pred_rew2 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj2], dim=0)
                pred_rank = torch.lt(pred_rew1, pred_rew2)
                gt_rank = torch.lt(rew1, rew2)
                acc_counter += torch.sum(pred_rank==gt_rank)
                counter += bs1
            else:
                reward1 = [reward_net(item) for item in traj1]  #???reward?????????
                reward2 = [reward_net(item) for item in traj2]
                for i in range(args.num_agent):
                    pred_rew1 = torch.cat([torch.sum(item[i], dim=0, keepdim=True) for item in reward1], dim=0)
                    pred_rew2 = torch.cat([torch.sum(item[i], dim=0, keepdim=True) for item in reward2], dim=0)
                    pred_rank = torch.lt(pred_rew1, pred_rew2)
                    gt_rank = torch.lt(rew1[:,i], rew2[:,i])
                    acc_counter[i] += torch.sum(pred_rank==gt_rank).cpu()
                    counter[i] += bs1
        print('Epoch {}, Acc {}'.format(epoch, np.array(acc_counter)/counter))
        if np.sum(np.array(acc_counter)/counter)/3 > best_acc:
            best_acc = np.sum(np.array(acc_counter)/counter)/3
            print("best accuracy:", best_acc)
            if args.network_mode == 'nnnsingle':
                torch.save(reward_net.state_dict(), 'checkpoints/{}_reward_net_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.env_name, args.mode, args.dataset_mode,args.network_mode,'agent'+str(args.agent), '_'.join([str(traj_num) for traj_num in args.train_traj_nums]) if args.train_traj_nums is not None else '', args.seed,'iter'+str(args.iter)))
            else:
                torch.save(reward_net.state_dict(), 'checkpoints/{}_reward_net_{}_{}_{}_{}_{}_{}.pth'.format(args.env_name, args.mode, args.dataset_mode,args.network_mode,'agent'+str(args.agent), args.seed,'iter'+str(args.iter)))
                #torch.save(reward_net.state_dict(), 'checkpoints/{}_reward_net_{}_{}_{}_{}_{}.pth'.format(args.env_name, args.mode, args.dataset_mode,args.network_mode, '_'.join([str(traj_num) for traj_num in args.train_traj_nums]) if args.train_traj_nums is not None else '', args.seed))

    for iter_, data in enumerate(train_loader):
        traj1, rew1, traj2, rew2 = data
        if use_gpu:
            traj1, rew1, traj2, rew2 = [item.cuda() for item in traj1], rew1.cuda(), [item.cuda() for item in traj2], rew2.cuda()
        bs1 = len(traj1)
        bs2 = len(traj2)
        assert bs1 == bs2
        '''for i in range(len(traj1)):  #1-64
            traj1[i][:,0] = traj1[i%16][:,0] #1/4
            rew1[i][0] = rew1[i%16][0]
            traj2[i][:,0] = traj2[i%16][:,0] #1/4
            rew2[i][0] = rew2[i%16][0]'''
        #pdb.set_trace()
        optimizer.zero_grad()
        if args.network_mode == 'nnnsingle':
            pred_rew1 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj1], dim=0)
            pred_rew2 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj2], dim=0)
            reward_sum = torch.cat([pred_rew1, pred_rew2], dim=1)
            rank_label = (torch.lt(rew1, rew2)).long()
            loss = nn.CrossEntropyLoss()(reward_sum, rank_label)
            loss.backward()
            optimizer.step()
            if iter_ % args.log_interval == 0:
                print('epoch {}, training loss {}'.format(epoch, loss.item()))
        else:
            loss = 0
            losslist = []#[0]*args.num_agent
            reward1 = [reward_net(item) for item in traj1]  #???reward?????????
            reward2 = [reward_net(item) for item in traj2]
            for i in range(args.num_agent):
                if iter_ < args.iter or (i+1)!= args.agent:
                    pred_rew1 = torch.cat([torch.sum(item[i], dim=0, keepdim=True) for item in reward1], dim=0)
                    pred_rew2 = torch.cat([torch.sum(item[i], dim=0, keepdim=True) for item in reward2], dim=0)
                    reward_sum = torch.cat([pred_rew1, pred_rew2], dim=1)
                    rank_label = (torch.lt(rew1[:,i], rew2[:,i])).long()
                    lossn = nn.CrossEntropyLoss()(reward_sum, rank_label)
                    losslist.append(lossn)
                    loss += lossn
            loss.backward()
            optimizer.step()
            if iter_ % args.log_interval == 0:
                print('epoch {}, training loss {}, seperate loss {}'.format(epoch, loss.item(), [x.item() for x in losslist]))


'''
\To change agent: change range in train_trex, and change range in models
sample call: 
python ./t_rex/train_trex.py --env-name simple_adversary --train_demo_files ./models/simple_adversary/torchversion/run3/1.pkl --test_demo_files ./models/simple_adversary/torchversion/run8/1.pkl --batch-size 64 --log-interval 100 --save-interval 10 --dataset_mode partial --num_epochs 1000 --batch-size 64 --mode state_only --trajdimension 6 --actiondimension 2 --network_mode single --seed 3

python ./t_rex/train_trex.py --env-name simple_adversary --train_demo_files ./models/simple_adversary/torchversion/run3/all.pkl --test_demo_files ./models/simple_adversary/torchversion/run8/all.pkl --batch-size 64 --log-interval 100 --save-interval 10 --dataset_mode partial --num_epochs 1000 --batch-size 64 --mode state_only --trajdimension 6 --actiondimension 2 --network_mode shared --seed 3

!python ./t_rex/train_trex.py --env-name simple_adversary --train_demo_files ./models/simple_adversary/withgoal/run1/allstate5025.pkl ./models/simple_adversary/withgoal/run2/allstate5025.pkl ./models/simple_adversary/withgoal/run3/allstate5025.pkl ./models/simple_adversary/withgoal/run4/allstate5025.pkl ./models/simple_adversary/withgoal/run5/allstate5025.pkl ./models/simple_adversary/withgoal/run6/allstate5025.pkl ./models/simple_adversary/withgoal/run7/allstate5025.pkl ./models/simple_adversary/withgoal/run8/allstate5025.pkl ./models/simple_adversary/withgoal/run9/allstate5025.pkl ./models/simple_adversary/withgoal/run10/allstate5025.pkl  ./models/simple_adversary/withgoal/run11/allstate5025.pkl ./models/simple_adversary/withgoal/run12/allstate5025.pkl ./models/simple_adversary/withgoal/run13/allstate5025.pkl ./models/simple_adversary/withgoal/run14/allstate5025.pkl ./models/simple_adversary/withgoal/run15/allstate5025.pkl --test_demo_files ./models/simple_adversary/withgoal/run16/allstate5025.pkl ./models/simple_adversary/withgoal/run17/allstate5025.pkl ./models/simple_adversary/withgoal/run18/allstate5025.pkl ./models/simple_adversary/withgoal/run19/allstate5025.pkl ./models/simple_adversary/withgoal/run20/allstate5025.pkl --batch-size 64 --log-interval 100 --save-interval 10 --dataset_mode partial --num_epochs 150 --batch-size 64 --mode state_only --trajdimension 15 --actiondimension 2 --network_mode shared --seed 5

!python ./t_rex/train_trex.py --env-name simple_adversary --train_demo_files ./models/simple_adversary/withgoal/run1/allstate25025.pkl ./models/simple_adversary/withgoal/run2/allstate25025.pkl ./models/simple_adversary/withgoal/run3/allstate25025.pkl ./models/simple_adversary/withgoal/run4/allstate25025.pkl ./models/simple_adversary/withgoal/run5/allstate25025.pkl ./models/simple_adversary/withgoal/run6/allstate25025.pkl ./models/simple_adversary/withgoal/run7/allstate25025.pkl ./models/simple_adversary/withgoal/run8/allstate25025.pkl ./models/simple_adversary/withgoal/run9/allstate25025.pkl ./models/simple_adversary/withgoal/run10/allstate25025.pkl ./models/simple_adversary/withgoal/run11/allstate25025.pkl ./models/simple_adversary/withgoal/run12/allstate25025.pkl ./models/simple_adversary/withgoal/run13/allstate25025.pkl ./models/simple_adversary/withgoal/run14/allstate25025.pkl ./models/simple_adversary/withgoal/run15/allstate25025.pkl --test_demo_files ./models/simple_adversary/withgoal/run16/allstate25025.pkl ./models/simple_adversary/withgoal/run17/allstate25025.pkl ./models/simple_adversary/withgoal/run18/allstate25025.pkl ./models/simple_adversary/withgoal/run19/allstate25025.pkl ./models/simple_adversary/withgoal/run20/allstate25025.pkl --batch-size 64 --log-interval 100 --save-interval 10 --dataset_mode partial --num_epochs 200 --batch-size 64 --mode state_only --trajdimension 15 --actiondimension 2 --network_mode shared --seed 5

!python ./t_rex/train_trex.py --env-name simple_adversary --train_demo_files ./models/simple_adversary/withgoal/run1/allstate5025.pkl ./models/simple_adversary/withgoal/run2/allstate5025.pkl ./models/simple_adversary/withgoal/run3/allstate5025.pkl ./models/simple_adversary/withgoal/run4/allstate5025.pkl ./models/simple_adversary/withgoal/run5/allstate5025.pkl ./models/simple_adversary/withgoal/run6/allstate5025.pkl ./models/simple_adversary/withgoal/run7/allstate5025.pkl ./models/simple_adversary/withgoal/run8/allstate5025.pkl ./models/simple_adversary/withgoal/run9/allstate5025.pkl ./models/simple_adversary/withgoal/run10/allstate5025.pkl  ./models/simple_adversary/withgoal/run11/allstate5025.pkl ./models/simple_adversary/withgoal/run12/allstate5025.pkl ./models/simple_adversary/withgoal/run13/allstate5025.pkl ./models/simple_adversary/withgoal/run14/allstate5025.pkl ./models/simple_adversary/withgoal/run15/allstate5025.pkl --test_demo_files ./models/simple_adversary/withgoal/run16/allstate5025.pkl ./models/simple_adversary/withgoal/run17/allstate5025.pkl ./models/simple_adversary/withgoal/run18/allstate5025.pkl ./models/simple_adversary/withgoal/run19/allstate5025.pkl ./models/simple_adversary/withgoal/run20/allstate5025.pkl --batch-size 64 --log-interval 100 --save-interval 10 --dataset_mode partial --num_epochs 150 --batch-size 64 --mode state_only --trajdimension 15 --actiondimension 2 --network_mode shared --seed 5

!python ./t_rex/train_trex.py --env-name simple_adversary --train_demo_files ./models/simple_adversary/withgoal/run1/all5025.pkl ./models/simple_adversary/withgoal/run2/all5025.pkl ./models/simple_adversary/withgoal/run3/all5025.pkl ./models/simple_adversary/withgoal/run4/all5025.pkl ./models/simple_adversary/withgoal/run5/all5025.pkl ./models/simple_adversary/withgoal/run6/all5025.pkl ./models/simple_adversary/withgoal/run7/all5025.pkl ./models/simple_adversary/withgoal/run8/all5025.pkl ./models/simple_adversary/withgoal/run9/all5025.pkl ./models/simple_adversary/withgoal/run10/all5025.pkl  --test_demo_files ./models/simple_adversary/withgoal/run11/all5025.pkl ./models/simple_adversary/withgoal/run12/all5025.pkl ./models/simple_adversary/withgoal/run13/all5025.pkl ./models/simple_adversary/withgoal/run14/all5025.pkl ./models/simple_adversary/withgoal/run15/all5025.pkl ./models/simple_adversary/withgoal/run16/all5025.pkl ./models/simple_adversary/withgoal/run17/all5025.pkl ./models/simple_adversary/withgoal/run18/all5025.pkl ./models/simple_adversary/withgoal/run19/all5025.pkl ./models/simple_adversary/withgoal/run20/all5025.pkl --batch-size 64 --log-interval 100 --save-interval 10 --dataset_mode partial --num_epochs 1000 --batch-size 64 --mode state_only --trajdimension 5 --actiondimension 2 --network_mode shared --seed 5

'''