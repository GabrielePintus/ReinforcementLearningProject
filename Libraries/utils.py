from typing import Any
from Libraries.data_handling import DataGenerator
from Libraries.agents import LearningAgent
from Libraries.tiling import TileEncodingApproximator, LCTC_ValueFunction
from Libraries.policies import Policy

import numpy as np
from tqdm.auto import tqdm




class RLProblem:

    @staticmethod
    def _from_dict(params):
        return RLProblem(**params)

    def __init__(
            self,
            data_path: str = None,
            levels: int = 1,
            horizon: int = 100,
            environment_class = None,
            phi_transform: callable = None,
            state_space_boundaries : list = None,
            agent_class: LearningAgent = None,
            value_function_class: TileEncodingApproximator = None,
            lctc_weights: np.ndarray = None,
            value_approx_params: dict = None,
            agent_params: dict = None,
            policy_class : Policy = None,
            policy_params: dict = None
        ):
        # Store the input arguments
        self.data_path = data_path
        self.levels = levels
        self.horizon = horizon
        self.environment_class = environment_class
        self.phi_transform = phi_transform
        self.state_space_boundaries = state_space_boundaries
        self.state_space_dims = [boundary.shape[0] for boundary in state_space_boundaries]
        self.agent_class = agent_class
        self.value_function_class = value_function_class
        self.lctc_weights = lctc_weights
        self.value_approx_params = value_approx_params
        self.agent_params = agent_params
        self.policy_class = policy_class
        self.policy_params = policy_params

        # Create data generator
        self.data_generator = DataGenerator(
            self.data_path,
            levels = self.levels,
            horizon = self.horizon+1,
            sequential = False
        )

        # Create Environment
        self.environment = self.environment_class(
            lob_data = self.data_generator[0],
            horizon = self.horizon,
            phi_transform = self.phi_transform,
        )

        # Create Value Function(s)
        value_functions = [
            self.value_function_class(
                state_dim = state_dim,
                bounds = bounds,
                **self.value_approx_params
            )
            for state_dim, bounds in zip(self.state_space_dims, self.state_space_boundaries)
        ]
        self.value_function = LCTC_ValueFunction(
            value_functions = value_functions,
            lctc_weights = self.lctc_weights
        )

        # Create Policy
        policy = self.policy_class(
            value_function = self.value_function,
            env = self.environment,
            **self.policy_params
        )

        # Create Agent
        self.agent = self.agent_class(
            env = self.environment,
            value_function = self.value_function,
            policy = policy,
            **self.agent_params
        )
        
    def get_values(self):
        raise NotImplementedError("This method is not implemented yet.")

    def train(
            self,
            n_scenarios: int = 10,
            episodes_per_scenario: int = 2,
            iterarations_per_episode: int = 1,
            verbose: bool = True
        ):
        self.last_train_params = {
            'n_scenarios': n_scenarios,
            'episodes_per_scenario': episodes_per_scenario,
            'iterarations_per_episode': iterarations_per_episode
        }

        # Create iterator
        n_iterations = n_scenarios * episodes_per_scenario
        iterator = range(n_iterations)
        if verbose:
            # progress bar
            iterator = tqdm(iterator)

        # Train the agent
        train_rewards, infos = dict(), dict()
        for i in iterator:
            # Compute scenario index
            idx = i % n_scenarios

            # Update the environment with the new scenario
            lob_data_scenario = self.data_generator[idx]
            self.environment.lob_data = lob_data_scenario

            # Train the agent
            _rewards, _infos = self.agent.train(n_episodes = iterarations_per_episode)

            # Store the rewards and infos
            train_rewards[idx] = train_rewards.get(idx, []) + [_rewards]
            infos[idx] = infos.get(idx, []) + [_infos]

            if verbose:
                iterator.set_postfix({
                    'Episode': i//n_scenarios,
                    'Epsilon': self.agent.policy.epsilon,
                })

        # convert rewards structure
        train_rewards = {k: np.vstack(v).reshape(episodes_per_scenario,iterarations_per_episode) for k, v in train_rewards.items()}


        # Return the rewards and infos
        return train_rewards, infos

    def test(
            self,
            n_scenarios: int = 10,
            epsilon: float = 0.0,
            oos : bool = True,
            verbose: bool = True,
        ):
        if not oos and n_scenarios > self.last_train_params['n_scenarios']:
            raise ValueError("Cannot test on-sample scenarios with more scenarios than trained on.")

        # Change epsilon for greedy policy
        self.agent.policy.epsilon = epsilon
        
        
        idxs = range(n_scenarios) if not oos else range(self.last_train_params['n_scenarios'], self.last_train_params['n_scenarios']+n_scenarios)
        if verbose:
            idxs = tqdm(idxs)

        rewards = dict()
        infos = dict()
        for idx in idxs:
            # Update the environment with the new scenario
            lob_data_scenario = self.data_generator[idx]
            self.environment.lob_data = lob_data_scenario

            # Test the agent
            rewards[idx], infos[idx] = self.agent.test(n_episodes = 1)

        return rewards, infos
            


class Metrics:

    # Profit and Loss function
    @staticmethod
    def PnL(phi_a, phi_b, inv, market_spread):
        return phi_a + phi_b + inv*market_spread
            
    # cumulative MAP
    @staticmethod
    def MAP(inv, prev_MAP, t):
        if(t == 1):
            return np.abs(inv)
        else:
            return (np.abs(inv) + prev_MAP*(t-1))/t

    
    
    
