import torch
from torch import nn
from torch.nn import functional as F



class MLP(nn.Module):
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation=nn.ReLU,
        dropout=0.0,
        batchnorm=False
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation()
        self.dropout = dropout
        self.batchnorm = batchnorm
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim[0]),
            self.activation
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_dim)):
            self.hidden_layers.append(nn.Dropout(self.dropout))
            self.hidden_layers.append(nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))
            if self.batchnorm:
                self.hidden_layers.append(nn.BatchNorm1d(self.hidden_dim[i]))
            self.hidden_layers.append(self.activation)
            
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim[-1], self.output_dim)
        )
        
        # Initialize weights
        self.init_weights()
        
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def init_weights(self):
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        

class Qnet(MLP):
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation=nn.ReLU,
        dropout=0.0,
        batchnorm=False,
        device = torch.device('cpu'),
        optimizer_params = dict(),
        criterion = nn.HuberLoss()
    ):
        super(Qnet, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            dropout,
            batchnorm
        )
        self.device = device
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **optimizer_params
        )
        self.to(self.device)