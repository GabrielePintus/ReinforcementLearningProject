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
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout
        self.device = device
        self._build_network()
        self.init_weights()

    def _build_network(self):
        self.layers = nn.ModuleList()
        # Add batch normalization to the input layer
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
    
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        




class QValueApproximator:

    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)#, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        self.data_loader = None
        self.n_epochs = 1
        self.batch_size = 40
        self.buffer = []
        self.loss = np.inf

    def predict(self, state):
        """
        Predict Q-values for a given state using the model.

        Parameters:
        - state (np.ndarray): The input state to predict Q-values for.

        Returns:
        - q_values (torch.Tensor): Predicted Q-values for all actions.
        """
        # Convert state to tensor and move to the appropriate device
        if not torch.is_tensor(state):
            state = torch.tensor(state).float().to(self.model.device)
        with torch.no_grad():  # No need to track gradients for prediction
            q_values = self.model(state.unsqueeze(0))  # Add batch dimension
        return q_values.squeeze(0)  # Remove batch dimension

    def update(self, y_true, y_pred):
        """
        Update the model using the provided target and predicted Q-values.

        Parameters:
        - y_true (torch.Tensor): Target Q-values.
        - y_pred (torch.Tensor): Predicted Q-values.

        Returns:
        - loss (float): The loss value.
        """
        # Check if input is tensor or numpy array
        if not torch.is_tensor(y_true):
            y_true = torch.tensor(y_true, requires_grad=True, dtype=torch.float32).to(self.model.device)
        if not torch.is_tensor(y_pred):
            y_pred = torch.tensor(y_pred, requires_grad=True, dtype=torch.float32).to(self.model.device)
        
        # Add to buffer
        self.buffer.append((y_pred, y_true))

        # Update model if buffer is full
        if len(self.buffer) == self.batch_size:
            self.data_loader = torch.utils.data.DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)
            self.buffer = []
        
            # Train the model
            for epoch in range(self.n_epochs):
                for y_pred, y_true in self.data_loader:
                    # Ensure data is on the correct device
                    y_pred = y_pred.to(self.model.device)
                    y_true = y_true.to(self.model.device)

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    loss = self.criterion(y_pred, y_true)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                    # Update weights
                    self.optimizer.step()
            
            self.loss = loss.item()



        return self.loss
