# SARSA for Windy Grid World


from collections import Counter
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

# world is a 10x7 grid
X = 10
Y = 7

epsilon = 0.1
alpha = 0.5

S = (0,3) # Source
G = (7,3) # Goal
W = [0,0,0,1,1,1,2,2,1,0] # Wind direction

A_4 = ['T', 'B', 'L', 'R']
A_9 = ['T', 'B', 'L', 'R', 'TL', 'TR', 'BL', 'BR', 'N'] # 9 actions including no movement
STOCHASTIC = False # Wind stochasticity

def get_next_state(state, action):
    x = state[0]
    y = state[1]
    if action == 'T':
        y += 1
    elif action == 'B':
        y -= 1
    elif action == 'L':
        x -= 1
    elif action == 'R':
        x += 1
    elif action == 'TL':
        x -= 1
        y += 1
    elif action == 'TR':
        x += 1
        y += 1
    elif action == 'BR':
        x += 1
        y -= 1
    elif action == 'BL':
        x -= 1
        y -= 1
    else:
        pass
    x, y = correct_state(x, y)
    return (x, y)

def correct_state(x, y):
    if x >= X:
        x = X - 1
    if x <= -1:
        x = 0
    if y >= Y:
        y = Y - 1
    if y <= -1:
        y = 0
    return x, y

def apply_wind(state, stochastic = False):
    x = state[0]
    y = state[1] + W[state[0]]
    if stochastic:
        r = random.random()
        if r < 1.0/3:
            y += 1
        elif r < 2.0/3:
            y -= 1
        else:
            pass
    x, y = correct_state(x, y)
    return (x, y)

def choose_action(state, q_values, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        state_action_values = map(lambda x: q_values[(state, x)], actions)
        max_value = max(state_action_values)
        max_indices = filter(lambda x: x[1] == max_value, enumerate(state_action_values))
        max_actions = map(lambda x: actions[x[0]], max_indices)
        return random.choice(max_actions)

def choose_best_action(state, q_values, actions):
    state_action_values = map(lambda x: q_values[(state, x)], actions)
    max_value = max(state_action_values)
    max_indices = filter(lambda x: x[1] == max_value, enumerate(state_action_values))
    max_actions = map(lambda x: actions[x[0]], max_indices)
    return random.choice(max_actions)


def sarsa(actions, wind_stochasticity):
    q_values = Counter()
    episodes = []
    print "Stochasticity of the wind: ", wind_stochasticity
    for i in tqdm(range(500)): # for 100 episodes
        cur_state = S
        cur_action = choose_action(cur_state, q_values, actions)
        time = 0
        while cur_state != G:
            time += 1
            next_state = get_next_state(cur_state, cur_action)
            next_state = apply_wind(next_state, wind_stochasticity)
            next_action = choose_action(next_state, q_values, actions)
            delta = -1 + q_values[(next_state, next_action)] - q_values[(cur_state, cur_action)]
            q_values[(cur_state, cur_action)] += alpha * delta
            cur_state = next_state
            cur_action = next_action
        episodes.extend([i]*time)
    return q_values, episodes


def get_path(q_values, actions, wind_stochasticity):
    state = S
    states = []
    while True:
        action = choose_best_action(state, q_values, actions)
        states.append((state, action))
        #print state, action, "current state, and action"
        if state == G:
            break

        state = get_next_state(state, action)
        state = apply_wind(state, wind_stochasticity)
    return states

if __name__ == '__main__':

    q_values_exm_6_5, episodes_exm_6_5 = sarsa(A_4, False)
    path_exm_6_5 = get_path(q_values_exm_6_5, A_4, False)
    print "Path Example 6.5: ", path_exm_6_5

    q_values_exr_6_7, episodes_exr_6_7 = sarsa(A_9, False)
    path_exr_6_7 = get_path(q_values_exr_6_7, A_9, False)
    print "Path Exercise 6.7: ", path_exr_6_7

    q_values_exr_6_8, episodes_exr_6_8 = sarsa(A_9, True)
    path_exr_6_8 = get_path(q_values_exr_6_8, A_9, True)
    print "Path Exercise 6.8: ", path_exr_6_8
    # The below plot shows how much time it took for an episode to terminate
    # as we progress further in the sarsa algorithm
    # the constant slope shows that the episode length stopped improving further,
    # it converged ~22 time steps
    # plt.figure()
    # plt.plot(episodes)
    # plt.xlabel('Time steps')
    # plt.ylabel('Episodes')
    # plt.show()
    f, (ax1, ax2, ax3) = plt.subplots(3, sharey=True, sharex=True)
    ax1.plot(episodes_exm_6_5)
    ax2.plot(episodes_exr_6_7)
    ax3.plot(episodes_exr_6_8)
    ax1.set_title('Example 6.5')
    ax2.set_title('Exercise 6.7')
    ax3.set_title('Exercise 6.8')
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()