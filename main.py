# Data Handling
import pandas as pd
import numpy as np
import pickle

# Custom Modules
from Libraries import environment as env
from Libraries import agents as ag
from Libraries import tiling
from Libraries.data_handling import DataGenerator
from Libraries.utils import RLProblem
from Libraries.policies import EpsilonGreedyPolicy

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set the seed for reproducibility
SEED = 42
np.random.seed(SEED)


# Get inputs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='')
parser.add_argument('--levels', type=int, default=5)
parser.add_argument('--reward', type=str, default='PnL')
parser.add_argument('--agent', type=str, default='SarsaLambda')
args = parser.parse_args()

# Reward transformation
if args.reward == 'PnL':
    reward_transform = env.PhiTransform.PnL()
elif args.reward == 'PnL_damp':
    reward_transform = env.PhiTransform.PnL_dampened(c=0.5)
elif args.reward == 'PnL_asymm_damp':
    reward_transform = env.PhiTransform.PnL_asymm_dampened(c=-0.5)
else:
    raise ValueError('Reward transformation not recognized')

# Agent
if args.agent == 'SarsaLambda':
    agent_class = ag.SarsaLambdaAgent
elif args.agent == 'QLambda':
    agent_class = ag.QLambdaAgent
else:
    raise ValueError('Agent not recognized')

# Extract stock name
stock_name = args.data_path.split('/')[-1].split('.')[0]
base_filename = f'output/{stock_name}_{args.levels}_{args.reward}_{args.agent}'



# Data Generator
data_generator = DataGenerator(args.data_path, levels=args.levels, horizon=120, sequential=False)
lob_data = data_generator[0]

# Boundaries
action_bounds = np.array([-1, 9])
# Compute state space bounds
agent_state_bounds = np.array([
    [-1e5, 1e5], # Inventory
    [0,2], # Spread
    [-1,5], # Theta_a
    [-1,5], # Theta_b
])
market_state_bounds = np.array([
    lob_data.min(axis=0) * 0.9,
    lob_data.max(axis=0) * 1.1,
]).T
full_state_bounds = np.vstack([agent_state_bounds, market_state_bounds])

# Add action bounds to each state
agent_state_bounds = np.vstack([agent_state_bounds, action_bounds])
market_state_bounds = np.vstack([market_state_bounds, action_bounds])
full_state_bounds = np.vstack([full_state_bounds, action_bounds])

# Group in a list
state_space_boundaries = [agent_state_bounds, market_state_bounds, full_state_bounds]

# Define the Linear Combination of Tile Codings weights
lctc_weights = np.array([0.6, 0.1, 0.3])

# RL Problem
rl_problem_params = {
    'data_path'         : args.data_path,
    'levels'            : args.levels,
    'horizon'           : 100,
    'environment_class' : env.Case3MarketMakerEnv,
    'phi_transform'     : reward_transform,
    'state_space_boundaries' : state_space_boundaries,
    'agent_class'       : agent_class,
    # 'agent_class'       : ag.QLambdaAgent,
    'value_function_class' : tiling.SparseTileEncodingApproximator,
    'lctc_weights'    : lctc_weights,
    'value_approx_params' : {
        'n_tiles' : 32,
        'n_tilings' : 32,
        'offset' : 0.17,
        'n_weights' : 16384,
    },
    'agent_params' : {
        'alpha' : 1e-3,
        'gamma' : 0.99,
        'el_decay' : 0.96,
    },
    'policy_class' : EpsilonGreedyPolicy,
    'policy_params' : {
        'epsilon' : 1.0,
        'epsilon_decay' : 0.9975,
        'epsilon_min' : 1e-1,
    },
}
rl_problem = RLProblem._from_dict(rl_problem_params)

# Train
train_rewards, train_infos = rl_problem.train(
    n_scenarios=1000,
    episodes_per_scenario=1,
    iterarations_per_episode=1,
    verbose=True
)
# save the trained agent
with open(base_filename+'_train_rewards.pkl', 'wb') as f:
    pickle.dump(train_rewards, f)
with open(base_filename+'_train_infos.pkl', 'wb') as f:
    pickle.dump(train_infos, f)

# Test
test_rewards, test_infos = rl_problem.test(
    n_scenarios=100,
    epsilon=0.0,
    oos=True,
    verbose=True
)
# save the test results
with open(base_filename+'_test_rewards.pkl', 'wb') as f:
    pickle.dump(test_rewards, f)
with open(base_filename+'_test_infos.pkl', 'wb') as f:
    pickle.dump(test_infos, f)


