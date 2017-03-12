
# State Aggregation
"""
Problem definition, 1000 states, from 1 to 1000
transition to one of the left 100 states or the right 100 states with equal probability
if there are less than 100 states on either side, the rest of the probability is for the terminal state.
"""

import random
from tqdm import tqdm
from matplotlib import pyplot as plt

TOTAL_STATES = 1000
alpha = 2*1e-5
TOTAL_EPISODES = 100000

STATES = range(TOTAL_STATES)
TERMINAL_STATES = set([0, TOTAL_STATES-1])
theta = [0]*10

def get_next_state(state):
    next_states = [state - x for x in range(1, 101)] + [state + x for x in range(1, 101)]
    next_states = map(lambda x: 0 if x < 0 else x, next_states)
    next_states = map(lambda x: TOTAL_STATES - 1 if x >= TOTAL_STATES else x, next_states)
    # print next_states, len(next_states)
    next_state = random.choice(next_states)
    if next_state == 0:
        reward = -1
    elif next_state == TOTAL_STATES - 1:
        reward = 1
    else:
        reward = 0
    return next_state, reward

def function_approx_score(state):
    group_index = state/100
    return theta[group_index]



# Gradient Monte Carlo algorithm

def generate_episode():
    state = 500
    episode = []
    while state not in TERMINAL_STATES:
        next_state, reward = get_next_state(state)
        episode.append((state, reward, next_state))
        state = next_state
    # episode.append((state, 0, state))
    return episode

def gradient_monte_carlo():
    theta = [0]*10
    for i in tqdm(range(TOTAL_EPISODES)):
        episode = generate_episode()
        final_reward = episode[-1][1]
        for time_step in episode:
            state, reward, next_state = time_step
            step_index = state/100
            if next_state in TERMINAL_STATES:
                theta[step_index] += alpha * (reward - theta[step_index])
            else:
                theta[step_index] += alpha * (reward + final_reward - theta[step_index])
    return theta

if __name__ == '__main__':
    theta = gradient_monte_carlo()

    # theta = [-1.035074104087505, -0.6609857751001176, -0.4805669042908276, -0.29707714042606176, -0.10500124638856906, 0.06981060050970521, 0.27036394304129036, 0.4565247132533697, 0.6542139836096942, 1.0345291545367545]
    values = map(function_approx_score, range(TOTAL_STATES))
    plt.plot(values)
    plt.show()
    print theta


