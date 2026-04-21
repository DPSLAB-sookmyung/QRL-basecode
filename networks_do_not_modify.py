import torch
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torchquantum as tq
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributions import normal
import pandas as pd
from collections import OrderedDict
from torchquantum.layers import U3CU3Layer0
import torch
import os
import math
# from shot_utils import get_data_loaders, gumbel_softmax

class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args     = args
        if args.mode == 'Classical' or args.mode == 'DQN' or args.mode == 'IQL':
            self.net      = nn.Sequential(nn.Linear(args.obs_dim, args.actor_dim) ,
                                      nn.ReLU(),
                                      nn.Linear(args.actor_dim, args.actor_dim) ,
                                      nn.ReLU(),
                                      nn.Linear(args.actor_dim, args.actor_dim) ,
                                      nn.ReLU(),
                                      nn.Linear(args.actor_dim, args.actor_dim) ,
                                      nn.ReLU(),
                                      nn.Linear(args.actor_dim, args.actor_dim) ,
                                      nn.ReLU(),
                                      nn.Linear(args.actor_dim, args.n_actions) ,
                                     )  # value function을 근사
        else:
            self.net      = QActor(args)
    
    def forward(self, inputs):
        q = self.net(inputs)
        return q

# Critic of Central-V
class CentralV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = nn.Sequential(
                  nn.Linear(args.obs_dim * args.n_agents, args.critic_dim),
                  nn.ReLU(),
                  nn.Linear(args.critic_dim, args.critic_dim),
                  nn.ReLU(),
                  nn.Linear(args.critic_dim, args.critic_dim),
                  nn.ReLU(),
                  nn.Linear(args.critic_dim, 1)
                                    )  # value function을 근사
        # if args.mode == 'Classical':
        #     self.net = nn.Sequential(
        #                           nn.Linear(args.obs_dim , args.critic_dim),
        #                           nn.ReLU(),
        #                           nn.Linear(args.critic_dim, args.critic_dim),
        #                           nn.ReLU(),
        #                           nn.Linear(args.critic_dim, args.critic_dim),
        #                           nn.ReLU(),
        #                           nn.Linear(args.critic_dim, 1)
        #                             )  # value function을 근사
        # else:
        #     self.net = QCritic(args)
    def forward(self, inputs):
        v = self.net(inputs)
        return v

# ========== Quantum Actor-Critic ========== #

class QActor(tq.QuantumModule):
    def __init__(self, args):  
        super().__init__()
        self.args       = args
        self.n_wires    = args.n_wires
        self.state_dim  = args.obs_dim
        self.action_dim = args.n_actions
        self.q_device   = tq.QuantumDevice(self.n_wires)
        self.get_encoder()
        self.get_PQC()
        self.measure   = tq.MeasureAll(tq.PauliZ)
        self.fc        = nn.Conv1d(1,1,1,1)
        
    def get_encoder(self):
        arch       = {'n_wires': self.n_wires, 'n_blocks': 1, 'n_layers_per_block': 1}
        self.reuploading_num = int(np.ceil(self.state_dim/16))
        encoderlist = []
        reuplist    = []
        for i in range(self.reuploading_num):
            start_idx = 16 * i 
            end_idx   = min(16 * (i+1), self.state_dim)
            encoderlist.append(tq.RLEncoder(self.n_wires, start_idx, end_idx))
            reuplist.append(U3CU3Layer0(arch))
        self.enc_data    = nn.ModuleList([*encoderlist])
        self.enc_reup    = nn.ModuleList([*reuplist])
        
    def get_PQC(self):
        arch       = {'n_wires': self.n_wires, 'n_blocks': 1, 'n_layers_per_block': 1}
        self.PQC    = U3CU3Layer0(arch)
        
        
    @tq.static_support
    def forward(self, data):
        bsz  = data.shape[0]
        data = data.reshape(bsz,-1)
        for i in range(self.reuploading_num):
            seg = data[:, 16*i:16*(i+1)]
            # 🔥 obs_dim이 16보다 작으면 zero-padding 추가
            if seg.shape[1] < 16:
                pad = torch.zeros(seg.shape[0], 16 - seg.shape[1], device=seg.device)
                seg = torch.cat([seg, pad], dim=1)
            self.enc_data[i](self.q_device, seg)
            self.enc_reup[i](self.q_device)
        self.PQC(self.q_device)
        if self.args.mode == 'QNN_PVM':
            value = self.q_device.get_states_1d().abs() ** 2  # 확률정책 
            # value += 1e-7
            # value = self.fc(torch.log(value))
        elif self.args.mode == 'QNN':
            value = self.measure(self.q_device)
            value = self.fc(value.unsqueeze(1)).squeeze() # value function을 근사
        return value  # value ~ [-1, +1] 
    

class QCritic(tq.QuantumModule):
    def __init__(self, args):  
        super().__init__()
        self.n_wires    = args.n_wires
        self.state_dim  = args.obs_dim
        self.action_dim = 1
        self.q_device   = tq.QuantumDevice(self.n_wires)
        self.get_encoder()
        self.get_PQC()
        self.measure   = tq.MeasureAll(tq.PauliZ)
        self.fc        = nn.Conv1d(1,1,1,1)
        
    def get_encoder(self):
        arch       = {'n_wires': self.n_wires, 'n_blocks': 1, 'n_layers_per_block': 1}
        self.reuploading_num = int(np.ceil(self.state_dim/self.n_wires))
        encoderlist = []
        reuplist    = []
        for i in range(self.reuploading_num):
            start_idx = 16 * i 
            end_idx   = min(16 * (i+1), self.state_dim)
            encoderlist.append(tq.RLEncoder(self.n_wires, start_idx, end_idx))
            reuplist.append(U3CU3Layer0(arch))
        self.enc_data    = nn.ModuleList([*encoderlist])
        self.enc_reup    = nn.ModuleList([*reuplist])
        
    def get_PQC(self):
        arch       = {'n_wires': self.n_wires, 'n_blocks': 3, 'n_layers_per_block': 3}
        self.PQC    = U3CU3Layer0(arch)
        
        
    @tq.static_support
    def forward(self, data):
        bsz  = data.shape[0]
        data = data.reshape(bsz,-1)
        for i in range(self.reuploading_num):
            self.enc_data[i](self.q_device, data[:,16*i:16*(i+1)])
            self.enc_reup[i](self.q_device)
        self.PQC(self.q_device)
        out = self.measure(self.q_device)[:,0]
        # print(out.shape)
        value = self.fc(out.reshape(-1,1,1)).squeeze() # value function을 근사
        return value  # value ~ [-1, +1] 