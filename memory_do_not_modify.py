import numpy as np
import collections
import random
from environment import EP_LEN
def get_experience(o,actions, rewards, o_prime, done):
    experience = {
            'o'       : o,
            'a'       : actions,
            'rewards' : rewards, 
            'o_prime' : o_prime,
            'done'    : done
        }
    return experience

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.buffer = [] 
    def push(self, experience):
        self.buffer.append(experience)
            
    def sample(self):
        O,ACTIONS, REWARDS, O_PRIME, DONE = [],[],[],[],[]
        for i in range(len(self.buffer)):
            O.append(self.buffer[i]['o'])
            ACTIONS.append(self.buffer[i]['a'])
            REWARDS.append(self.buffer[i]['rewards'])
            O_PRIME.append(self.buffer[i]['o_prime'])
            DONE.append(self.buffer[i]['done'])
        O = np.array(O)
        ACTIONS = np.array(ACTIONS)
        REWARDS = np.array(REWARDS)
        O_PRIME = np.array(O_PRIME)
        DONE    = np.array(DONE)

        samples = dict({
            'O' : O,
            'A' : ACTIONS,
            'REWARDS' : REWARDS,
            'O_PRIME' : O_PRIME,
            'DONE' : DONE
        })
        self.buffer = [] 
        return samples

    def size(self):
        return len(self.buffer)

# import numpy as np
# import threading
# import collections

# def get_experience(o,actions, rewards, o_prime, done):
#     experience = {
#             'o': o,
#             'a' : actions,
#             'rewards': rewards, 
#             'o_prime': o_prime,
#             'done': done,
#         }
#     return experience
# class ReplayBuffer:
#     def __init__(self, args):
#         self.args = args
#         self.buffer = collections.deque(maxlen = self.args.replay_capacity)
        
#         # store the episode
#     def push(self, experience):
#         self.buffer.append(experience)
#         # if len(self.buffer) > self.args.replay_capacity:
#         #     del self.buffer[0]
            
#     def sample(self):
#         # idx = np.arange(len(self.buffer))
#         # np.random.shuffle(idx)
#         # idx=idx[:self.args.batch_size]
#         O, ACTIONS, REWARDS, O_PRIME, DONE = [],[],[],[],[]
#         for i in range( self.size() ):
#             O.append(self.buffer[i]['o'])
#             ACTIONS.append(self.buffer[i]['a'])
#             REWARDS.append(self.buffer[i]['rewards'])
#             O_PRIME.append(self.buffer[i]['o_prime'])
#             DONE.append(self.buffer[i]['done'])
#         O = np.array(O)
#         ACTIONS = np.array(ACTIONS)
#         REWARDS = np.array(REWARDS)
#         O_PRIME = np.array(O_PRIME)
#         DONE    = np.array(DONE)
#         samples = dict({
#             'O' : O,
#             'A' : ACTIONS,
#             'REWARDS' : REWARDS,
#             'O_PRIME' : O_PRIME,
#             'DONE' : DONE,
#         })
        
#         return samples

#     def size(self):
#         return len(self.buffer)