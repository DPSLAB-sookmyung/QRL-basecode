import numpy as np

# ============================================================
# ITRC Quantum RL Environment (MISO EE Optimization)
#  - Base Station: 1 (16 antennas, 1 RF chain)
#  - User: 1 (fixed)
#  - Action: (beam index, power index)
#  - State: [quantized SNR, prev_beam, prev_power, prev_EE]
#  - Reward: α·EE − β·beam_switch − γ·power_switch
# ============================================================

# ---------- Simulation Parameters ----------
EP_LEN        = 100        # Episode length
EPOCH         = 15000
N_AGENTS      = 1           # Single BS
N_BEAMS       = 11          # Beam candidates (DFT codebook)
N_POWERS      = 5           # Power levels

BANDWIDTH     = 100e6       # 100 MHz
NOISE_DENSITY = 3.98e-21    # W/Hz
NOISE_POWER   = BANDWIDTH * NOISE_DENSITY
P_CIRCUIT     = 0.2         # Circuit power (Baseband+RF+Phase Shifter)
P_MAX         = 1.0         # Max TX power
P_LEVELS      = np.array([0.1, 0.25, 0.5, 0.75, 1.0]) * P_MAX

ALPHA, BETA, GAMMA = 0.5, 0.2, 0.1  # Reward weights

# ---------- Helper Functions ----------

def dft_codebook(num_antennas=16):
    """Generate DFT beamforming codebook (num_antennas x num_beams)."""
    n = np.arange(num_antennas)
    codebook = []
    for k in range(num_antennas):
        w = (1/np.sqrt(num_antennas)) * np.exp(-1j * 2 * np.pi * n * k / num_antennas)
        codebook.append(w)
    return np.array(codebook)

def compute_snr(h, w, p_tx):
    """Compute received SNR given channel h, beam w, and transmit power p_tx."""
    return np.abs(np.vdot(h, w))**2 * p_tx / NOISE_POWER

# def compute_ee(snr, p_tx):
#     """Compute energy efficiency [bit/J]."""
#     return BANDWIDTH * np.log2(1 + snr) / (p_tx + P_CIRCUIT)

def compute_ee(snr, p_tx):
    """Compute energy efficiency [bit/J]."""
    # 기존: BANDWIDTH * log2(1 + snr) / (p_tx + P_CIRCUIT)
    ee = BANDWIDTH * np.log2(1 + snr) / (p_tx + P_CIRCUIT)
    return ee / 1e9

def quantize_snr(snr, n_levels=10):
    """Quantize SNR for discrete state representation."""
    snr_db = 10 * np.log10(snr + 1e-12)
    snr_db = np.clip(snr_db, -5, 30)
    bins = np.linspace(-5, 30, n_levels)
    return np.digitize(snr_db, bins) / n_levels  # normalized 0~1

# ---------- Core Classes ----------

class Agent:
    def __init__(self):
        self.Nt = 16
        self.codebook = dft_codebook(self.Nt)
        self.h = (np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)) / np.sqrt(2)
        self.prev_beam = 0
        self.prev_power = 0
        self.prev_ee = 0.0
        self.last_snr = 1e-3

    def step(self, beam_index, power_index):
        """Perform one transmission step and return reward, next state, etc."""
        p_tx = P_LEVELS[power_index]
        w = self.codebook[beam_index]
        snr = compute_snr(self.h, w, p_tx)
        ee = compute_ee(snr, p_tx)

        # Reward components
        beam_switch = int(beam_index != self.prev_beam)
        power_switch = abs(power_index - self.prev_power) / N_POWERS
        reward = (ALPHA * ee - BETA * beam_switch - GAMMA * power_switch) / 1e7

        # Update internal state
        self.prev_beam = beam_index
        self.prev_power = power_index
        self.prev_ee = ee
        self.last_snr = snr

        # Construct next state (PDF-based definition)
        state = np.array([
            quantize_snr(snr),
            beam_index / N_BEAMS,
            power_index / N_POWERS,
            ee / 1e8,  # normalized EE
        ], dtype=np.float32)

        return state, reward, ee, snr

class Env:
    def __init__(self):
        self.t = 0
        self.agent = Agent()
        self.rewards = []
        self.EE_log = []
        self.SNR_log = []

    def reset(self):
        self.t = 0
        self.agent = Agent()
        self.rewards.clear()
        self.EE_log.clear()
        self.SNR_log.clear()
        # initial state
        init_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return init_state

    def step(self, action):
        """action = (beam_index, power_index)"""
        beam_index, power_index = action
        next_state, reward, ee, snr = self.agent.step(beam_index, power_index)

        self.rewards.append(reward)
        self.EE_log.append(ee)
        self.SNR_log.append(snr)

        self.t += 1
        done = (self.t >= EP_LEN)
        return next_state, reward, done, {}

    def get_episode_stats(self):
        """Return average EE, average SNR, total reward."""
        return {
            "avg_EE": np.mean(self.EE_log) if self.EE_log else 0.0,
            "avg_SNR": np.mean(self.SNR_log) if self.SNR_log else 0.0,
            "total_reward": np.sum(self.rewards) if self.rewards else 0.0,
        }

# ---------- Example Run ----------
if __name__ == "__main__":
    env = Env()
    state = env.reset()
    print(f"[Init state] {state}")

    for t in range(EP_LEN):
        beam = np.random.randint(0, N_BEAMS)
        power = np.random.randint(0, N_POWERS)
        state, reward, done, _ = env.step((beam, power))
        if done:
            break

    stats = env.get_episode_stats()
    print("\n=== Episode Summary ===")
    print(f"Total reward: {stats['total_reward']:.4e}")
    print(f"Average EE  : {stats['avg_EE']:.4e}")
    print(f"Average SNR : {stats['avg_SNR']:.4e}")
