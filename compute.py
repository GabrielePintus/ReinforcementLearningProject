import os

# Get all stocks
files = [ str(file) for file in os.listdir('data') if file.endswith('.parquet') ]
stocks = [ file.split('.')[0] for file in files ]
stocks = sorted(stocks)

# Levels
levels = [1, 5, 10]

# Reward functions
rewards = ['PnL', 'PnL_damp', 'PnL_asymm_damp']

# Agents
agents = ['SarsaLambda', 'QLambda']

# Create the grid
from itertools import product
grid = list(product(stocks, levels, rewards, agents))

# Create the command list
commands = []
for stock, level, reward, agent in grid:
    command = f'sbatch run.sbatch {stock} {level} {reward} {agent}'
    commands.append(command)

# Write the commands to a file
with open('commands.txt', 'w') as f:
    for command in commands:
        f.write(f'{command}\n')
