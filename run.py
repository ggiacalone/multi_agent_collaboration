from collections import deque
import numpy as np
import torch

def run(env, agent, n_episodes=2000, max_t=1000, print_every=100):

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores_deque = deque(maxlen=print_every)

    scores = []
    scores1 = []
    scores2 = []

    for i_episode in range(1, n_episodes+1):

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]

        # get the current state
        states = env_info.vector_observations

        agent.reset()

        score1 = 0
        score2 = 0

        # while True:
        for t in range(max_t):

            actions = agent.act(states)

            # send the action to the environment
            env_info = env.step(actions)[brain_name]

            # get the next state
            next_states = env_info.vector_observations

            # get the reward
            rewards = env_info.rewards

            # see if episode has finished
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            score1 += rewards[0]
            score2 += rewards[1]

            if np.any(dones):
                break

        score = max(score1, score2)
        scores_deque.append(score)
        scores.append(score)

        scores1.append(score1)
        scores2.append(score2)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) > 0.5:

            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-print_every, np.mean(scores_deque)))

            agent.save_checkpoint()

            break

    return scores, scores1, scores2
