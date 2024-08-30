from Libraries import agents
import pandas as pd


def train_agent(
        agent : agents.LearningAgent = None,
        n_episodes : int = 10
    ):
    # Check the input arguments
    assert agent is not None
    assert n_episodes > 0

    # Train the agent
    train_rewards, infos = agent.train(n_episodes)
    df_infos = pd.DataFrame(infos)

