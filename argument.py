from easydict import EasyDict as edict
from environment import N_BEAMS, N_POWERS, N_AGENTS, EP_LEN
import torchquantum as tq

# ============================================================
# Argument setup for MISO Quantum RL
# ============================================================

def get_args():
    args = edict({
        'mode'                : 'QNN_PVM',   # MonteCarlo, Classical, QNN_PVM, QNN, DQN, IQL
        'seed'                : 43,
        'target_update_cycle' : 200,        # 예: 200 스텝/업데이트마다 하드 동기화
        'epsilon'             : 0.3,
        'anneal_epsilon'      : 5e-5,
        'min_epsilon'         : 0.01,
        'replay_capacity'     : 50000,
        'batch_size'          : 64,
        'actor_dim'           : 32,
        'critic_dim'          : 256,
        'train_epoch'         : 15000,
        'cuda'                : True,
        'gamma'               : 0.98,
        'actor_lr'            : 1e-3,
        'critic_lr'           : 2.5e-4,
        'use_dueling'         : False,
        'use_per'             : False,
        # 'epsilon'             : 0.275,
        # 'anneal_epsilon'      : 0.00005,
        # 'min_epsilon'         : 0.01,
        # 'replay_capacity'     : 50000,
        # 'target_update_cycle' : 20,
        'grad_norm_clip'      : 10,
        'k'                   : 2,
        'train_size'          : 2000,
        # Quantum parameters
        'n_wires'             : N_BEAMS,    # 16개 빔 = 16 큐비트 (beam encoding)
        'n_pqc'               : 1,
    })

    args.q_device = tq.QuantumDevice(n_wires=args.n_wires)
    return args


def get_env_info(args, env=None):
    """
    환경 정보 자동 반영 (QRL용)
    """
    args.n_agents      = N_AGENTS       # 1 (BS)
    args.n_actions = 2 ** 16
    args.obs_dim       = 4              # [SNR, prev_beam, prev_power, prev_EE]
    args.episode_limit = EP_LEN         # 100 steps
    args.n_actions_beam = 11
    args.n_actions_power = 5
    return args
