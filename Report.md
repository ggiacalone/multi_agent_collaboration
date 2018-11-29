# Report

### Learning Algorithm

This is an implementation of a Multi-Agent DDPG algorithm. I used the code from the ddqn-pendulumn project as a starting point, which is located here: https://github.com/udacity/deep-reinforcement-learning/tree/master/ddqn-pendulum, and modified it slightly to account for multiple agents.

DDPG, is a model-free, off-policy actor-critic algorithm that uses deep neural networks as function approximinators to both  learn an action policy (the actor), and to evaluate the expected reward of that action policy (the critic).

In this Multi-Agent implementation, both agents share the same actor/crtic network, and a shared experiences memory.

### Model Architecture

The model aritechture uses an Actor/Critic setup.

The Actor is a simple neural network consisting of 3 fully connected layers. The model takes as input a vector of the state space (33 states), runs it through a first layer of 400 connected units, then a second layer of 300 units again, and finally through a classification layer that outputs a vector of probabilities the respective action space (2 in this case).

The Critic is a simple neural network with the same architecture, only it outputs a single continuous value.

### Hyper parameters

I started out with the default parameters in the starter code:  
BUFFER_SIZE = int(1e5) # replay buffer size  
BATCH_SIZE = 128 # minibatch size  
GAMMA = 0.99 # discount factor  
TAU = 1e-3 # for soft update of target parameters  
LR_ACTOR = 1e-4 # learning rate  
LR_CRITIC = 1e-3  
WEIGHT_DECAY = 0  
  
UPDATE_EVERY = 4 # how often to update the network  
  
### Submission parameters:

The best result was acheived using the following hyperparameters:  
  
BUFFER_SIZE = int(1e6)  # replay buffer size  
BATCH_SIZE = 512
GAMMA = 0.99            # discount factor  
TAU = 0.01              # for soft update of target parameters  
LR_ACTOR = 1e-4         # learning rate of the actor   
LR_CRITIC = 1e-4        # learning rate of the critic  
WEIGHT_DECAY = 0
UPDATE_EVERY = 10        # how often to update the network  
  
You can look at `Explore.ipynb` for a list of the experiments I tried. I first experimented with a MADDPG implementation where each agent used its own actor/crtic network however I was unable to get it to learn. You can see this experiment in code cell 22, and the MADDPG implementaton in the file `maddpg.py`.

As suggested in the student hub and in the baseline implementaton notes I next tried to use a shared actor/critic for all agents (just 2 in this project), and also used a shared memory replay buffer. This implementation is in the file `maddpg_agent.py` With some changes in hyperparameters I was finally able to meet the environment solve requirements.

You can see a plot of rewards for my final submission in the last code cell of `Explore.ipynb`.

### Implementation notes:

I also implemented gradient clipping as described in the project description, and only update the network weights every 4 timestep's to control learning stability.

### Ideas for Future Work:
- Try solving the Soccer environment.
- Try using Prioritized Experience Replay.
- Try to implement MA4PG.

### Submission Saved Model Weights
`maddpg_checkpoint_actor.pth`
`maddpg_checkpoint_critic.pth`

