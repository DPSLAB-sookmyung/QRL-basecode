import numpy as np
import torch
from torch.distributions import Categorical
from networks_do_not_modify import Actor, CentralV
from copy import deepcopy
from torch.nn.functional import one_hot
import torch.nn as nn

class Agent:
    def __init__(self, args, id = None):
        self.id        = id
        self.args      = args
        self.n_actions = args.n_actions
        # self.state_shape = args.state_dim
        self.obs_shape = args.obs_dim
        self.actor        = Actor(args)
        if self.args.mode == 'DQN' : self.actor_target = deepcopy(self.actor)
        self.actor.cuda()
        if self.args.mode == 'DQN' : self.actor_target.cuda()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr) 
        
    def A_preprocessing(self, A):
        return A[:,self.id]
    
    def O_preprocessing(self, O):
        return  torch.FloatTensor(O[:,self.id,:])

    def o_preprocessing(self, o):
        agent_obs   = torch.FloatTensor(o)[self.id]
        return agent_obs.unsqueeze(0)
                
    def choose_action_from_softmax(self, action_dist):
        prob   = torch.nn.functional.softmax(action_dist, dim = -1)
        action = torch.argmax(prob, -1)
        return action
    
    def select_action(self, o):
        agent_obs   = self.o_preprocessing(o)
        action_dist = self.actor(agent_obs.cuda())
        if self.args  != 'QNN_PVM':
            action      = self.choose_action_from_softmax(action_dist)
        else:
            action      = torch.argmax(action_dist, -1)
        return action
    
    # CY added #
    def train(self,O,A,RWD,O_PRIME,critic):
        obs       = O
        obs_prime = O_PRIME
        O         = self.O_preprocessing(O)
        O_PRIME   = self.O_preprocessing(O_PRIME)
        O_C       = self.O_preprocessing(obs)
        O_C_PRIME = self.O_preprocessing(obs_prime)
        O         = torch.FloatTensor(O).cuda()
        O_PRIME   = torch.FloatTensor(O_PRIME).cuda()
        O_C       = torch.FloatTensor(O_C).cuda()
        O_C_PRIME = torch.FloatTensor(O_C_PRIME).cuda()
        r         = torch.FloatTensor(RWD).squeeze().cuda().unsqueeze(-1)
        A         = torch.tensor(A.reshape(-1,1),dtype=torch.int64).cuda()
        Q         = self.actor(O).gather(dim=1, index=A)
        TD_TARGET = r + self.args.gamma * critic.critic(O_C_PRIME)
        DELTA     = TD_TARGET - critic.critic(O_C)
        loss      = - torch.log(Q) * DELTA.detach() + nn.MSELoss()(critic.critic(O_C), TD_TARGET.detach())

        self.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        loss.mean().backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), self.args.grad_norm_clip)
        critic.optimizer.step()

        return loss.mean().item()
    

class Critic:
    def __init__(self,args):
        self.args      = args
        self.critic    = CentralV(args).cuda()
        self.optimizer = torch.optim.Adam(self.critic.parameters(),lr=args.critic_lr)
        
