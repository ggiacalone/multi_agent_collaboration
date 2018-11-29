import numpy as np
import random
import copy
from collections import namedtuple, deque

from noise import OUNoise
from memory import ReplayBuffer
from model import Actor, Critic

# https://github.com/rlcode/per
# from prioritized_memory import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size

BATCH_SIZE = 512        # minibatch size
# BATCH_SIZE = 64        # minibatch size
# BATCH_SIZE = 1024

GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
TAU = .01

# LR_ACTOR = 1e-4         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic
# LR_CRITIC = 3e-4        # learning rate of the critic

# LR_ACTOR = 2e-4         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic

LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic

WEIGHT_DECAY = 0        # L2 weight decay
# WEIGHT_DECAY = 0.0001   # L2 weight decay
# WEIGHT_DECAY = 0.001


UPDATE_EVERY = 10        # how often to update the network

NUM_AGENTS = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MaddpgAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = [OUNoise(action_size, random_seed) for i in range(NUM_AGENTS)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # self.soft_update(self.critic_local, self.critic_target, 1)
        # self.soft_update(self.actor_local, self.actor_target, 1)

    def save_experiences(self, states, actons, rewards, next_states, dones):
        self.memory.add(states, actons, rewards, next_states, dones)

    def step(self, states, actions, rewards, next_states, dones):

        for i in range(NUM_AGENTS):
            # Save experience / reward
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])


        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for i in range(UPDATE_EVERY):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):

        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:

            for i in range(NUM_AGENTS):
                action = actions[i]
                for j in action:
                    j += self.noise[i].sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        for i in range(NUM_AGENTS):
            self.noise[i].reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # use gradient clipping
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_checkpoint(self):
        torch.save(self.actor_local.state_dict(), 'maddpg_checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'maddpg_checkpoint_critic.pth')
