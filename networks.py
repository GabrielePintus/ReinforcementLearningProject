import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Maybe we can tinker with optimizer parameters and network architecture, 
# since Pusher is a "simple" environment

class ActorNetwork(nn.Module):
    '''
    ActorNetwork class defines the actor network for the SAC algorithm, which is responsible for the approximation of the policy function.
    '''
    def __init__(self,
                lr : float,
                input_dim : np.array,
                n_actions : int, 
                max_action : float, 
                hidden1_dim : int = 256, 
                hidden2_dim : int = 256,
                name : str ='actor',
                chkpt_dir : str ='tmp/sac'): 
        '''
        Constructor for the ActorNetwork class.
        lr: the learning rate.
        input_dims: the dimension of the input data.
        max_action: the maximum action value, it is used to obtain the appropriate action interval from the tanh activation function
        hidden1_dim: the number of neurons in the first hidden layer.
        hidden2_dim: the number of neurons in the second hidden layer.
        n_actions: the number of possible actions.
        name: str, the name of the network.
        chkpt_dir: str, the directory to save the network's checkpoints.
        '''
    
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action

        # Reparameterization trick noise, used to avoid numerical instability 
        # and avoid log(0) in the log probability calculation
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dim, self.hidden1_dim)
        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.mu = nn.Linear(self.hidden2_dim, self.n_actions)
        self.sigma = nn.Linear(self.hidden2_dim, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        '''
        Forward pass of the network.
        state: the current state.
        return: the mean and standard deviation of the action distribution.
        '''
        
        # Correct to use same network for both? Paper implies so

        distr = self.fc1(state)
        distr = F.relu(distr)
        distr = self.fc2(distr)
        distr = F.relu(distr)

        mu = self.mu(distr)
        sigma = self.sigma(distr)
        
        # Clamping the standard deviation to avoid numerical instability
        # Small diff from paper implementation, here we avoid 0 sigma. Check if it's correct
        # Could use directly a softplus (log(1+exp(x))) to ensure positive values
        # Also why 0,1 and not 0,2 or 0,3? To control exploration?
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        '''
        Sample actions from the normal distribution.
        state: the current state.
        reparameterize: whether to reparameterize the action.
        return: the action and the log probability of the action.
        '''

        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)


        # Don't we always want to reparameterize to add some noise?
        # Reparameterization trick: z = mu + sigma * epsilon where epsilon ~ N(0, 1)

        # if reparameterize:
        actions = probabilities.rsample()
        # else:
             #actions = probabilities.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)

        # This is for the loss calculation
        log_probs = probabilities.log_prob(actions)

        # Need to better understand why we do this, check paper appendix...
        log_probs = log_probs - torch.log(1-action.pow(2)+self.reparam_noise)

        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        '''
        Save the network's checkpoint.
        '''
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        '''
        Load the network's checkpoint.
        '''
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    '''
    CriticNetwork class defines the critic network for the SAC algorithm, which is responsible for the approximation the Q-value function.
    '''

    def __init__(self, input_dim : np.array,
                       n_actions : int,
                       lr : float = 5e-4,
                       hidden1_dim : int = 256, # paper dims
                       hidden2_dim : int = 256, # paper dims 
                       name : str = 'critic',
                       chkpt_dir : str = 'tmp/sac'):
        '''
        Constructor for the CriticNetwork class.
        input_dim: the dimension of the input data.
        n_actions:  the number of possible actions.
        lr: the learning rate.
        hidden1_dim: the number of neurons in the first hidden layer.
        hidden2_dim: the number of neurons in the second hidden layer.
        name: the name of the network.
        chkpt_dir: the directory to save the network's checkpoints.
        '''
        super(CriticNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # We're passing the state and action pairs
        self.fc1 = nn.Linear(self.input_dim + n_actions, self.hidden1_dim)
        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.q = nn.Linear(self.hidden2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        '''
        Forward pass of the network.
        state: the current state.
        action: the action taken.
        return: the Q-value of the state-action pair.
        '''

        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        '''
        Save the network's checkpoint.
        '''
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        '''
        Load the network's checkpoint.
        '''
        self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    '''
    ValueNetwork class defines the value network for the SAC algorithm, which is responsible for the approximation of the value state function.
    '''
    def __init__(self,
                input_dim : np.array,
                lr : float = 5e-4,
                hidden1_dim : int = 256,
                hidden2_dim : int = 256,
                name : str = 'value',
                chkpt_dir : str = 'tmp/sac'):
        '''
        Constructor for the ValueNetwork class.
        input_dim: the dimension of the input data.
        lr: the learning rate.
        hidden1_dim: the number of neurons in the first hidden layer.
        hidden2_dim: the number of neurons in the second hidden layer.
        name: str, the name of the network.
        chkpt_dir: str, the directory to save the network's checkpoints.
        '''

        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dim, self.hidden1_dim)
        self.fc2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.v = nn.Linear(self.hidden2_dim, 1) # output is a scalar

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        '''
        Forward pass of the network.
        state: the current state.
        return: the value of the state.
        '''
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        '''
        Save the network's checkpoint.
        '''
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        '''
        Load the network's checkpoint.
        '''
        self.load_state_dict(torch.load(self.checkpoint_file))
