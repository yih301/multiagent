import torch
import torch.nn as nn
import torchvision.models as pretrain_models
import torch.nn.functional as F

activation = nn.LeakyReLU

class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super(RewardNet, self).__init__()
        self.input_dim = input_dim
        last_dim = self.input_dim
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(last_dim, hidden_dim))
            layer_list.append(activation())
            last_dim = hidden_dim
        layer_list.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.net(x)


    def compute_reward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.size()) == 1:
                x = x.view(1, -1)
            reward = self.net(x)
        return reward.item()   

class RewardNets(nn.Module):
    def __init__(self, input_dim, net_num, hidden_dim=256, num_layers=4):
        super(RewardNets, self).__init__()
        self.reward_nets = nn.ModuleList([RewardNet(input_dim, hidden_dim, num_layers) for i in range(net_num)])
        self.net_num = net_num

    def forward(self, x_list):
        if type(x_list) == list and len(x_list) == self.net_num:
            return [self.reward_nets[i](x_list[i]) for i in range(self.net_num)]
        else:
            return [self.reward_nets[i](x_list) for i in range(self.net_num)]

    def compute_reward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.size()) == 1:
                x = x.view(1, -1)
            reward = np.sum([self.reward_nets[i](x).item() for i in range(self.net_num)])
        return reward

class ShareRewardNet(nn.Module):
    def __init__(self, input_dim, net_num=3, hidden_dim=256, num_sharelayers=2, num_layers=2):
        super(ShareRewardNet, self).__init__()
        self.input_dim = input_dim
        last_dim = self.input_dim
        nets = []
        share1 = nn.Linear(input_dim, hidden_dim)
        share2 = nn.Linear(hidden_dim, hidden_dim)
        for i in range(net_num):
            layer_list = []
            layer_list.append(share1)
            layer_list.append(activation())
            layer_list.append(share2)
            layer_list.append(activation())
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(activation())
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(activation())
            layer_list.append(nn.Linear(hidden_dim, 1))
            nets.append(nn.Sequential(*layer_list))
        self.reward_nets = nn.ModuleList(nets)
        self.net_num = net_num

    def forward(self, x_list):
        if type(x_list) == list and len(x_list) == self.net_num:
            return torch.FloatTensor([self.reward_nets[i](x_list[i]) for i in range(self.net_num)],requires_grad=True)
        else:
            return torch.FloatTensor([self.reward_nets[i](x_list) for i in range(self.net_num)],,requires_grad=True)