import random

from ddpg_agent import Agent
from memory import ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024
UPDATE_EVERY = 3        # how often to update the network
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

class MaddpgAgent():

    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):

        self.agents = [Agent(state_size=state_size, action_size=action_size, random_seed=random_seed),
                      Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)]

        self.seed = random.seed(random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # self.soft_update(self.critic_local, self.critic_target, 1)
        # self.soft_update(self.actor_local, self.actor_target, 1)

    def act(self, states, add_noise=True):
        actions = [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):

        # Shared replay buffer
        for i, _ in enumerate(self.agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def learn(self, experiences, gamma):
        for agent in self.agents:
            agent.learn(experiences, gamma)


    def reset(self):
        for agent in self.agents:
            agent.reset()

    def save_checkpont(self):
        for i, agent in enumerate(self.agents):
            agent.save_checkpont(i)
