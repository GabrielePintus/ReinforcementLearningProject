import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MLP(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        activation=nn.ReLU,
        dropout=0.0,
        device=torch.device('cpu')
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout
        self.device = device
        self._build_network()

    def _build_network(self):
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        self.layers.append(self.activation)
        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(nn.Dropout(self.dropout))
            self.layers.append(nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.layers.append(self.activation)
        self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.softmax(x)
        




class QValueApproximator:

    def __init__(self, model):
        self.model = model

    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.model(state).numpy()
        
    def update(self, state, target):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
        # NN Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999))
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        output = self.model(state)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        return loss.item()

    def batch_update(self, states, targets):
        states = torch.tensor(states, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(states, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # NN Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999))
        criterion = nn.MSELoss()
        for state, target in loader:
            optimizer.zero_grad()
            output = self.model(state)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        return loss.item()
    
    
    
    
    
        
