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
import math

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

    # CSV 헤더: 요청하신 모든 항목
    DICT = {
        'epoch': [], 'reward': [], 'EE': [], 'SNR': [],
        'loss_total': [], 'actor_loss': [], 'critic_loss': [], 'entropy': [],
        'epsilon': [], 'V_mean': [], 'TD_target_mean': [],
        'grad_actor': [], 'grad_critic': [],
        'lr_actor': [], 'lr_critic': []
    }

    for ep in range(EPOCH):
        state = env.reset()
        Reward, EE_list, SNR_list = [], [], []

        with torch.no_grad():
            for t in range(EP_LEN):
                if np.random.rand() < args.epsilon:
                    # ε-greedy: random
                    beam_idx  = np.random.randint(0, args.n_actions_beam)
                    power_idx = np.random.randint(0, args.n_actions_power)
                else:
                    obs_t = torch.FloatTensor(state).unsqueeze(0).cuda()
                    q_values = agent.actor(obs_t)  # [1, n_actions_beam + n_actions_power (분리 사용)]
                    beam_logits  = q_values[:, :args.n_actions_beam]
                    power_logits = q_values[:, args.n_actions_beam: args.n_actions_beam + args.n_actions_power]
                    beam_idx  = torch.argmax(torch.softmax(beam_logits,  dim=-1)).item()
                    power_idx = torch.argmax(torch.softmax(power_logits, dim=-1)).item()

                next_state, reward, done, _ = env.step((beam_idx, power_idx))
                Reward.append(reward)
                EE_list.append(env.EE_log[-1])
                SNR_list.append(env.SNR_log[-1])
                replay.push(get_experience(state, (beam_idx, power_idx), reward, next_state, done))
                state = next_state
                if done:
                    break

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

        # actor 출력
        q_values = agent.actor(O)  # [bsz, n_actions_beam + n_actions_power (분리 해석)]
        beam_logits  = q_values[:, :args.n_actions_beam]
        power_logits = q_values[:, args.n_actions_beam: args.n_actions_beam + args.n_actions_power]

        # 분포 및 엔트로피 (샘플 평균)
        beam_prob  = torch.softmax(beam_logits,  dim=-1)
        power_prob = torch.softmax(power_logits, dim=-1)
        # 엔트로피 H(p) = -sum p log p
        # (log에 작은 값 더해 underflow 방지)
        eps = 1e-8
        beam_entropy  = -(beam_prob  * (beam_prob.clamp_min(eps)).log()).sum(dim=-1).mean()
        power_entropy = -(power_prob * (power_prob.clamp_min(eps)).log()).sum(dim=-1).mean()
        entropy_mean  = (beam_entropy + power_entropy).item()

        # policy loss (log-prob of taken actions)
        logp_beam  = torch.log_softmax(beam_logits,  dim=-1).gather(1, A_beam.view(-1, 1))
        logp_power = torch.log_softmax(power_logits, dim=-1).gather(1, A_power.view(-1, 1))
        policy_log = (logp_beam + logp_power)  # [bsz, 1]

        # critic
        V = critic.critic(O.reshape(O.shape[0], -1))                       # [bsz, 1]
        V_PRIME = critic.critic(O_PRIME.reshape(O_PRIME.shape[0], -1)).detach()
        TD_TARGET = RWD + args.gamma * V_PRIME
        advantage = TD_TARGET - V

        # 손실 분리
        actor_loss  = -(policy_log * advantage.detach()).mean()
        critic_loss = torch.nn.MSELoss()(V, TD_TARGET)
        loss_total  = actor_loss + critic_loss

        # 최적화 & grad norm
        agent.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        loss_total.backward()
        # clip 함수는 "클리핑 전" 전체 노름을 리턴
        grad_actor_norm  = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),  args.grad_norm_clip)
        grad_critic_norm = torch.nn.utils.clip_grad_norm_(critic.critic.parameters(), args.grad_norm_clip)
        agent.optimizer.step()
        critic.optimizer.step()

        # 통계값
        total_rwd = float(np.sum(Reward))
        avg_EE, avg_SNR = float(np.mean(EE_list)), float(np.mean(SNR_list))
        V_mean = float(V.mean().item())
        TD_target_mean = float(TD_TARGET.mean().item())
        lr_actor = float(agent.optimizer.param_groups[0].get('lr', 0.0))
        lr_critic = float(critic.optimizer.param_groups[0].get('lr', 0.0))

        print(
            f"[Epoch {ep}] Reward={total_rwd:.3e} | EE={avg_EE:.3e} | SNR={avg_SNR:.3e} | "
            f"loss_total={loss_total.item():.4f} (actor={actor_loss.item():.4f}, critic={critic_loss.item():.4f}) | "
            f"entropy={entropy_mean:.3f} | eps={args.epsilon:.4f} | "
            f"V_mean={V_mean:.3e} | TD_target_mean={TD_target_mean:.3e} | "
            f"grad_actor={float(grad_actor_norm):.3e} | grad_critic={float(grad_critic_norm):.3e} | "
            f"lr_a={lr_actor:.2e} | lr_c={lr_critic:.2e}"
        )

        # CSV 기록
        DICT['epoch'].append(ep)
        DICT['reward'].append(total_rwd)
        DICT['EE'].append(avg_EE)
        DICT['SNR'].append(avg_SNR)
        DICT['loss_total'].append(float(loss_total.item()))
        DICT['actor_loss'].append(float(actor_loss.item()))
        DICT['critic_loss'].append(float(critic_loss.item()))
        DICT['entropy'].append(float(entropy_mean))
        DICT['epsilon'].append(float(args.epsilon))
        DICT['V_mean'].append(V_mean)
        DICT['TD_target_mean'].append(TD_target_mean)
        DICT['grad_actor'].append(float(grad_actor_norm))
        DICT['grad_critic'].append(float(grad_critic_norm))
        DICT['lr_actor'].append(lr_actor)
        DICT['lr_critic'].append(lr_critic)

        # 매 에포크 저장
        pd.DataFrame(DICT).to_csv(f"{log_dir}/training_info.csv", index=False)

    print("=== Training Finished ===")

if __name__ == "__main__":
    main()
