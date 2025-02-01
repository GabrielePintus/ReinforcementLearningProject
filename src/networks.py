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
        
    def forward(self, x):
        if self.batchnorm:
            x = x.view(-1, self.input_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def init_weights(self, random=True):
        if random:
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.1)
                    nn.init.constant_(layer.bias, 0)
            self.input_layer[0].weight.data.normal_(0, 0.1)
            self.input_layer[0].bias.data.fill_(0)
            self.output_layer[0].weight.data.normal_(0, 0.1)
            self.output_layer[0].bias.data.fill_(0)
            
        else:
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            self.input_layer[0].weight.data.fill_(0)
            self.input_layer[0].bias.data.fill_(0)
            self.output_layer[0].weight.data.fill_(0)
            self.output_layer[0].bias.data.fill_(0)
            
        

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
        optimizer_params = { 'lr': 1e-3 },
        lr_scheduler_params = { 'gamma': 0.9 },
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
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **optimizer_params
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            **lr_scheduler_params
        )
        
        # Move the network to the device
        self.to(self.device)
        


class ActorNet(MLP):
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation=nn.ReLU,
        dropout=0.0,
        batchnorm=False,
        device = torch.device('cpu'),
        optimizer_params = { 'lr': 1e-3 },
        lr_scheduler_params = { 'gamma': 0.9 },
        output_activation = nn.Softmax
    ):
        super(ActorNet, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            dropout,
            batchnorm
        )
        self.output_activation = output_activation(dim=-1)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **optimizer_params
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            **lr_scheduler_params
        )
        
        # Move the network to the device
        self.to(self.device)
        
    def forward(self, x):
        x = super(ActorNet, self).forward(x)
        return self.output_activation(x)

    

class CriticNet(MLP):
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation=nn.ReLU,
        dropout=0.0,
        batchnorm=False,
        device = torch.device('cpu'),
        optimizer_params = { 'lr': 1e-3 },
        lr_scheduler_params = { 'gamma': 0.9 },
    ):
        super(CriticNet, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            dropout,
            batchnorm
        )
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **optimizer_params
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            **lr_scheduler_params
        )
        
        # Move the network to the device
        self.to(self.device)
        
    def forward(self, x):
        return super(CriticNet, self).forward(x)
    
    
    