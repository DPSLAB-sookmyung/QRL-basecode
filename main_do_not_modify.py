from argument import get_args, get_env_info
from environment import Env
from agent_do_not_modify import Agent,Critic
from memory_do_not_modify import ReplayBuffer, get_experience
import torch
import copy
import random
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from environment import N_AGENTS,N_OBSERVER,EP_LEN,EPOCH,N_REGION
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train(O,A,RWD,O_PRIME,agent,critic,mode):
    
    if mode != 'IQL': 
        bsz = O.shape[0]
        args = critic.args
        Q         = agent.actor(O[:, agent.id])
        Q         = Q.gather(dim=1, 
                             index=A.unsqueeze(1))
        V         = critic.critic(O.reshape(bsz,-1)) # bsz, n_agents * obs_dim current: V(S) <- previous: V(O)
        V_PRIME   = critic.critic(O_PRIME.reshape(bsz,-1)).detach()
        TD_TARGET = RWD + args.gamma * V_PRIME
        DELTA     = TD_TARGET - V
        loss      = - torch.log(Q) * DELTA.detach() + torch.nn.MSELoss()(V, TD_TARGET)

        agent.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        loss.mean().backward()

        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.grad_norm_clip)
        agent.optimizer.step()
        agent.optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), args.grad_norm_clip)
        critic.optimizer.step()
        critic.optimizer.zero_grad()
    
    else: # mode == 'IQL'
        pass
    
    return loss.mean().abs().item()

        
    # # Critic Train #
    # critic.optimizer.zero_grad()
    # # bsz      = O.shape[0]
    # critic_O = torch.FloatTensor(agent.O_preprocessing(O)).cuda()
    # critic_O_prime = torch.FloatTensor(agent.O_preprocessing(O_PRIME)).cuda()
    # R         = torch.FloatTensor(RWD).squeeze().cuda().unsqueeze(-1)
    # TD_TARGET = R.squeeze() + critic.args.gamma * critic.critic(critic_O_prime)
    # V         = critic.critic(critic_O)
    # Critic_loss = torch.nn.MSELoss()(V, TD_TARGET.detach())
    # Critic_loss.backward()
    # torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), args.grad_norm_clip)
    # critic.optimizer.step()

    # # Actor Train #
    # DELTA     = (TD_TARGET - V).detach()
    # A         = torch.tensor(A.reshape(-1,1), dtype = torch.int64).cuda()
    # agent.optimizer.zero_grad()
    # O_agent   = torch.FloatTensor(agent.O_preprocessing(O)).cuda()
    # # O_PRIME_agent   = torch.FloatTensor(agent.O_preprocessing(O_PRIME)).cuda()
    # # A_agent   = agent.A_preprocessing(A)
    # Q         = agent.actor(O_agent).gather(dim = 1, index = A)
    # actor_loss  = (- torch.log(Q) * DELTA).mean()
    # actor_loss.backward()
    # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.grad_norm_clip)
    # agent.optimizer.step()
        
    # return actor_loss.item(), Critic_loss.item()

def main():
    DICT = {'epoch': [],
            'reward': [],
            'utility': [],
            'cost': [],
            'AoI': [],
            'popularity': []}
    env = Env()
    args = get_args()
    args = get_env_info(args, env)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    today = datetime.today().strftime("%m%d%H%M%S")
    info  = f'mode{args.mode}_alr{args.actor_lr}_alr{args.critic_lr}'
    writer = SummaryWriter(f'runs/action_space_{args.n_actions}/{today}+{info}')

    replay = ReplayBuffer(args)
    REWARD, COST, UTILITY, A_LOSS, C_LOSS = [], [], [], [], []
    agents = [Agent(args, i) for i in range(N_AGENTS)]
    critic = Critic(args)
    for ep in range(EPOCH):
        TB_RWD     = 0
        TB_UTILITY = 0
        TB_COST    = 0
        o, _, _, _ = env.reset()
        Reward     = [] 
        with torch.no_grad():
            for t in range(EP_LEN-1):
                actions = []
                for i in range(N_AGENTS):
                    coin = np.random.rand()
                    if coin >= args.epsilon:
                        action = agents[i].select_action(o).squeeze().data.cpu().numpy()
                    else:
                        action = np.random.randint(0, 2**(N_REGION*N_OBSERVER))
                    actions.append(action)
                actions = np.array(actions)
                o, rewards, o_prime, done = env.step(actions)
                if done: break
                experience = get_experience(o, actions, rewards, o_prime, done)
                replay.push(experience)
                Reward.append(rewards)

        args.epsilon -= args.anneal_epsilon
        args.epsilon  = max(args.epsilon, args.min_epsilon)
        
        samples = replay.sample()
        O       = samples['O']
        ACTIONS = samples['A']
        REWARDS = samples['REWARDS']
        O_PRIME = samples['O_PRIME']
        DONE    = samples['DONE'] 
        A_LOSS, C_LOSS = [],[]
        
        O         = torch.FloatTensor(O).cuda() # bsz, n_agents, obs_dim
        O_PRIME   = torch.FloatTensor(O_PRIME).cuda() # bsz, n_agents, obs_dim
        REWARDS   = torch.FloatTensor(REWARDS).squeeze().cuda().unsqueeze(-1)  # bsz, 1
        ACTIONS   = torch.tensor(ACTIONS,dtype=torch.int64).cuda() # bsz, n_agents (2^16, 0~65535 사이 숫자 갖고있음)
        if args.mode != 'IQL':
            for i, agent in enumerate(agents):
                loss = train(O,ACTIONS[:,i],REWARDS,O_PRIME,agent,critic,args.mode)
                A_LOSS.append(loss)
                # C_LOSS.append(critic_loss)
                
        else: # args.mode == 'IQL'
            for i, agent in enumerate(agents):
                loss = agent.train(O,ACTIONS[:,i],REWARDS,O_PRIME,agent,critic,args.mode)
                A_LOSS.append(loss)
            
        REWARD.append(np.array(Reward))
        print(f'[Epoch {ep} | {args.mode}] \
                 Reward: {np.array(Reward).sum()}, \
                 Cost: {np.array(env.Cost).sum()}, \
                 Utility: {np.array(env.Utility).sum()}, \
                 AoI:{np.array(env.ActionAoI).mean()}, \
                 Popularity:{np.array(env.Popularity).mean()},')
        COST.append(env.Cost)
        UTILITY.append(env.Utility)
        DICT['epoch'].append(ep)
        DICT['reward'].append(np.array(Reward).sum())
        DICT['cost'].append(np.array(env.Cost).sum())
        DICT['utility'].append(np.array(env.Utility).sum())
        DICT['AoI'].append(np.array(env.ActionAoI).sum())
        DICT['popularity'].append(np.array(env.Popularity).sum())
        writer.add_scalars(f'Reward/reward', {
            'reward'  : np.array(Reward).sum(),
            'utility' : np.array(env.Cost).sum(),
            'cost'    : np.array(env.Utility).sum(),
            'AoI'    : np.array(env.ActionAoI).sum(),
            'popularity'    : np.array(env.Popularity).sum(),
        }, ep)
        pd.DataFrame(DICT).to_csv(f'runs/action_space_{args.n_actions}/{today}+{info}/training_info.csv')
    REWARD = np.array(REWARD)
    COST = np.array(COST)
    UTILITY = np.array(UTILITY)
    A_LOSS = np.array(A_LOSS)
    C_LOSS = np.array(C_LOSS)
    np.save(f'runs/{today}+{info}/REWARD.npy' , arr=REWARD)
    np.save(f'runs/{today}+{info}/COST.npy'   , arr=COST)
    np.save(f'runs/{today}+{info}/UTILITY.npy', arr=UTILITY)
    np.save(f'runs/{today}+{info}/A_LOSS.npy' , arr=A_LOSS)
    np.save(f'runs/{today}+{info}/C_LOSS.npy' , arr=C_LOSS)
        
if __name__ == "__main__":
    main()

# for epoch in range(args.train_epoch):
#     env.reset()
#     o         = env.o
#     o_prime   = env.o_prime
#     ava       = env.get_available_action()
#     TB_RWD    = np.zeros(args.n_agents)
#     TB_SR     = 0
#     TB_RES    = np.zeros(args.n_agents)
#     TB_OL     = 0
#     TB_EN     = np.zeros(args.n_agents)
    
#     for t in range(args.episode_limit+1):
#         o = o_prime.copy()
#         actions, u_onehot = [], np.zeros((args.n_agents,args.n_actions))
#         for i in range(args.n_agents):
#             coin = np.random.rand()
#             if coin >= args.epsilon:
#                 action = Agents[i].select_action(o=o, ava=ava[i]).squeeze().data.cpu().numpy()
#             else:
#                 action = np.random.randint(low=0,high=args.n_actions)
#             actions.append(action)
#         actions = np.array(actions)
#         o, rewards, o_prime, done = env.step(actions, t, epoch, today, info)
#         experience = get_experience(o, actions, rewards, o_prime, done)
#         replay.push(experience)
#         TB_RWD += rewards
#         sr, sa, TB_RES, ol, TB_EN = env.utility.get_utils_info()
#         TB_SR += sr
#         TB_OL += ol
#         # if (epoch+1) % 500 == 0 or (epoch == 0):
#         #     if t % 5 == 0:
#         #         image = env.plot()
#         #         writer.add_image(f'trajectory 2D/EP{epoch+1}', image, t)
#         #     writer.add_scalars(f'num_user/EP{epoch+1}', {
#         #     'total' : env.utility.SUPPORT[0],
#         #     'agent1': env.utility.SUPPORT[1],
#         #     'agent2': env.utility.SUPPORT[2],
#         #     'agent3': env.utility.SUPPORT[3],
#         #     'agent4': env.utility.SUPPORT[4]
#         #     }, t)
#         #     writer.add_scalars(f'energy_consumption/EP{epoch+1}', {
#         #         'agent1': TB_EN[0],
#         #         'agent2': TB_EN[1],
#         #         'agent3': TB_EN[2],
#         #         'agent4': TB_EN[3],
#         #     }, t)
#         #     writer.add_scalars(f'quality/EP{epoch+1}', {
#         #     'total' : TB_RES.sum(),
#         #     'agent1': TB_RES[0],
#         #     'agent2': TB_RES[1],
#         #     'agent3': TB_RES[2],
#         #     'agent4': TB_RES[3]
#         # }, t)

#     # [1] Sampling Batch Experiences
#     args.epsilon -= args.anneal_epsilon
#     args.epsilon  = max(args.epsilon, args.min_epsilon)
#     if replay.size() > args.train_size:
#         samples = replay.sample()
#         O = samples['O']
#         ACTIONS = samples['A']
#         REWARDS = samples['REWARDS']
#         O_PRIME = samples['O_PRIME']
#         ACTOR_LOSS = []
        
#         for i in range(args.n_agents):
#             loss = Agents[i].train(O,ACTIONS[:,i],REWARDS[:,i],O_PRIME,Central_V)
#             ACTOR_LOSS.append(loss)

#         writer.add_scalars(f'loss', {
#         'agent1': ACTOR_LOSS[0],
#         'agent2': ACTOR_LOSS[1],
#         'agent3': ACTOR_LOSS[2],
#         'agent4': ACTOR_LOSS[3],
#         }, epoch)

#     ###############################################################

#     # writer.add_scalars(f'reward', {
#     #         'total' : TB_RWD.sum(),
#     #         'agent1': TB_RWD[0],
#     #         'agent2': TB_RWD[1],
#     #         'agent3': TB_RWD[2],
#     #         'agent4': TB_RWD[3],
#     #     }, epoch)
#     # writer.add_scalars(f'utility/support_rate', {
#     #         'average' : TB_SR / args.episode_limit
#     #     }, epoch)
#     # writer.add_scalars(f'utility/quality', {
#     #         'total' : TB_RES.sum(),
#     #         'agent1': TB_RES[0],
#     #         'agent2': TB_RES[1],
#     #         'agent3': TB_RES[2],
#     #         'agent4': TB_RES[3],
#     #     }, epoch)
#     # writer.add_scalars(f'utility/overlapped', {
#     #         'average' : TB_OL/ args.episode_limit
#     #     }, epoch)

#     if epoch % 500 == 0:
#         for i in range(args.n_agents):
#             torch.save({
#                         'actor': Agents[i].actor.state_dict(),
#                         'actor_optimizer': Agents[i].actor_optimizer.state_dict()
#                         }, f'./Model/actor{i}_epoch{epoch}.tar')
        
#         torch.save({
#                     'critic': Central_V.critic.state_dict(),
#                     'critic_optimizer': Central_V.critic_optimizer.state_dict()
#                     }, f'./Model/critic_epoch{epoch}.tar')