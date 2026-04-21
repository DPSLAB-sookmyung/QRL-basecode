from argument import get_args, get_env_info
from environment import Env, EP_LEN, EPOCH
from agent_do_not_modify import Agent, Critic
from memory_do_not_modify import ReplayBuffer, get_experience
import torch
import numpy as np
import random
import pandas as pd
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def train(O, A, RWD, O_PRIME, agent, critic, mode, ep):
    if ep == 0:
        print(f"[DEBUG] O={O.shape}, A={A.shape}, RWD={RWD.shape}")
    args = critic.args
    loss_fn = nn.MSELoss()
    # batch 크기 동기화
    bsz = min(O.shape[0], A.shape[0], RWD.shape[0], O_PRIME.shape[0])
    O, A, RWD, O_PRIME = O[:bsz], A[:bsz], RWD[:bsz], O_PRIME[:bsz]
    # Value function
    V = critic.critic(O.reshape(bsz, -1))
    V_PRIME = critic.critic(O_PRIME.reshape(bsz, -1)).detach()
    # TD target & advantage
    TD_TARGET = RWD + args.gamma * V_PRIME
    DELTA = TD_TARGET - V
    # Policy network
    Q_all = agent.actor(O[:, agent.id])  # [bsz, n_actions]
    policy = torch.softmax(Q_all, dim=-1)
    log_prob = torch.log(policy.gather(1, A.view(-1, 1)) + 1e-8)
    # Actor-Critic loss
    actor_loss = -(log_prob * DELTA.detach()).mean()
    critic_loss = loss_fn(V, TD_TARGET)
    loss = actor_loss + critic_loss
    # Gradient step
    agent.optimizer.zero_grad()
    critic.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.grad_norm_clip)
    torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), args.grad_norm_clip)
    agent.optimizer.step()
    critic.optimizer.step()
    return loss.item()

def main():
    env = Env()
    args = get_args()
    args = get_env_info(args, env)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    today = datetime.today().strftime("%m%d%H%M%S")
    info = f'mode{args.mode}_alr{args.actor_lr}_clr{args.critic_lr}'
    log_dir = f'runs/MISO_QRL/{today}+{info}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    replay = ReplayBuffer(args)
    agent = Agent(args, 0)
    critic = Critic(args)

    DICT = {'epoch': [], 'reward': [], 'EE': [], 'SNR': [], 'loss': []}

    for ep in range(EPOCH):
        state = env.reset()
        Reward, EE_list, SNR_list = [], [], []

        with torch.no_grad():
            for t in range(EP_LEN):
                if np.random.rand() < args.epsilon:
                    # 🎯 무작위 선택
                    beam_idx  = np.random.randint(0, args.n_actions_beam)
                    power_idx = np.random.randint(0, args.n_actions_power)
                else:
                    obs_t = torch.FloatTensor(state).unsqueeze(0).cuda()
                    q_values = agent.actor(obs_t)  # [1, 65536]
                    # 🎯 앞쪽 일부를 beam으로, 다음 일부를 power로 사용
                    beam_logits  = q_values[:, :args.n_actions_beam]
                    power_logits = q_values[:, args.n_actions_beam: args.n_actions_beam + args.n_actions_power]

                    beam_idx  = torch.argmax(torch.softmax(beam_logits, dim=-1)).item()
                    power_idx = torch.argmax(torch.softmax(power_logits, dim=-1)).item()

                next_state, reward, done, _ = env.step((beam_idx, power_idx))
                Reward.append(reward)
                EE_list.append(env.EE_log[-1])
                SNR_list.append(env.SNR_log[-1])
                replay.push(get_experience(state, (beam_idx, power_idx), reward, next_state, done))
                state = next_state
                if done: break

        # Epsilon decay
        args.epsilon = max(args.min_epsilon, args.epsilon - args.anneal_epsilon)

        # === 학습 ===
        samples = replay.sample()
        A_np = np.array(samples['A'])
        A_beam  = torch.tensor(A_np[:, 0], dtype=torch.int64).cuda()
        A_power = torch.tensor(A_np[:, 1], dtype=torch.int64).cuda()

        O = torch.FloatTensor(samples['O']).cuda()
        RWD = torch.FloatTensor(samples['REWARDS']).cuda().unsqueeze(-1)
        O_PRIME = torch.FloatTensor(samples['O_PRIME']).cuda()

        # actor 출력 해석 동일
        q_values = agent.actor(O)  # [bsz, 65536]
        beam_logits  = q_values[:, :args.n_actions_beam]
        power_logits = q_values[:, args.n_actions_beam: args.n_actions_beam + args.n_actions_power]

        # policy 손실 beam/power 각각 계산
        policy_beam  = torch.log_softmax(beam_logits, dim=-1).gather(1, A_beam.view(-1, 1))
        policy_power = torch.log_softmax(power_logits, dim=-1).gather(1, A_power.view(-1, 1))
        policy_log   = policy_beam + policy_power

        # critic
        V = critic.critic(O.reshape(O.shape[0], -1))
        V_PRIME = critic.critic(O_PRIME.reshape(O_PRIME.shape[0], -1)).detach()
        TD_TARGET = RWD + args.gamma * V_PRIME
        advantage = TD_TARGET - V

        loss = -(policy_log * advantage.detach()).mean() + torch.nn.MSELoss()(V, TD_TARGET)

        agent.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.grad_norm_clip)
        torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), args.grad_norm_clip)
        agent.optimizer.step()
        critic.optimizer.step()

        total_rwd = np.sum(Reward)
        avg_EE, avg_SNR = np.mean(EE_list), np.mean(SNR_list)
        print(f"[Epoch {ep}] Reward={total_rwd:.3e} | EE={avg_EE:.3e} | SNR={avg_SNR:.3e} | loss={loss.item():.4f}")
        DICT['epoch'].append(ep)
        DICT['reward'].append(total_rwd)
        DICT['EE'].append(avg_EE)
        DICT['SNR'].append(avg_SNR)
        DICT['loss'].append(loss.item())
        pd.DataFrame(DICT).to_csv(f"{log_dir}/training_info.csv", index=False)

    print("=== Training Finished ===")

if __name__ == "__main__":
    main()