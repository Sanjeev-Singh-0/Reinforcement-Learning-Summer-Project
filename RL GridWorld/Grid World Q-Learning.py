
import numpy as np
import random
import matplotlib.pyplot as plt

# 5 x 5 grid 
n_x = 5
n_y = 5

environment = np.zeros((n_y, n_x))

initial_state = (0,0)

loc_positive_reward = (4,4)
loc_negative_reward = [(2,2),(2,3),(3,2),(3,3)]

# Assign each cell of grid a reward
def rewards(state):
    state_y, state_x = state
    if state == loc_positive_reward:
        reward = 100
    elif state in loc_negative_reward:
        reward = -30
    else:
        reward = -1
    return reward

# Visual check to ensure environment has been set up correctly
rewards_field = np.zeros_like(environment)

for i in range(0,n_y):
    for j in range(0, n_x):
        rewards_field[i,j] = rewards((i,j))

print(rewards_field)



# Define actions that agent (robot) can take

actions = {0: (-1,0), # up
           1: (1,0), # down
           2: (0,-1), # left
           3: (0,1), #right
           }

n_actions = 4 # used when generating the Q table



# Assigning index 0 to 24 to each of the cell on the grid
def map_state_to_index(state, n_x):
    state_y, state_x = state
    return state_y * n_x + state_x


beta = 0.7 # learning rate
gamma = 0.9 # discount factor
epsilon = 1.0 # probability of taking a random non optimal action
epsilon_decay = 0.999
min_epsilon = 0.01
num_trials = 300 # number of trial to train robot

# Q table 


Q_table = np.empty((n_y * n_x, n_actions))

# Create a Q table with random values between 100 and 200

for j in range(n_x*n_y):
    for i in range(n_actions):
        val = random.randint(100,200)
        Q_table[j][i] = val


# Empty array to store reward values after each trial
store_tot_reward = np.array([])
store_steps_taken = np.array([])

for trial in range(num_trials):

    #Set initial conditions
    state = initial_state
    total_reward = 0
    steps_taken = 0

    while state != loc_positive_reward:

        state_index = map_state_to_index(state, n_x)

        if random.random() < epsilon:
            action = random.choice(list(actions.keys()))
        else:
            action = np.argmax(Q_table[state_index])


        move = actions[action]
        next_state = (state[0] + move[0],state[1] + move[1])

        next_state = (
            max(0, min(n_y - 1, next_state[0])),
            max(0, min(n_x - 1, next_state[1]))
        )

        reward = rewards(next_state)


        total_reward = total_reward + reward

        next_state_index = map_state_to_index(next_state, n_x)

        Q_observed = reward + gamma * np.max(Q_table[next_state_index]) # Bellman Equation where gamma is the discount factor
        tde = Q_observed - Q_table[state_index][action] # Temporal difference error
        Q_table[state_index][action] = Q_table[state_index][action] + beta * tde # Update rule

        # Use an epsilon greedy function where epsilon decays with each trial
        epsilon = max(min_epsilon , epsilon * epsilon_decay)

        state = next_state

        steps_taken = steps_taken + 1

    store_tot_reward = np.append(store_tot_reward,total_reward) 
    store_steps_taken = np.append(store_steps_taken,steps_taken)




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot of total reward against the number of trials
ax1.plot(store_tot_reward)
ax1.set_title('Total Reward')

# Plot of steps taken against the number of trials
ax2.plot(store_steps_taken)
ax2.set_title('Steps Taken')

plt.tight_layout()
plt.savefig('Results.png')
plt.show()


print(Q_table)

print(store_tot_reward)
print(store_steps_taken)
