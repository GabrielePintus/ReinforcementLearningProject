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

    def __init__(self, model, learning_rate=1e-4, std_values=np.array):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)#, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        self.data_loader = None
        self.n_epochs = 3
        self.batch_size = 40
        self.buffer = []
        self.loss = np.inf
        self.std_values = torch.tensor(std_values, dtype=torch.float32).to(self.model.device)

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
            # print("State before", state)
            state = (state - self.std_values[:, 0]) / self.std_values[:, 1]
            # print("State after", state)
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





from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError

class TilingApproximatorMedium:

    def __init__(self, bounds, n_tiles, n_tilings, shifts):
        self.bounds = bounds
        self.n_tiles = n_tiles
        self.n_tilings = n_tilings
        self.shifts = shifts
        self.model = SGDRegressor(
            max_iter=1,
            tol=None,
            penalty='l2',
            alpha=1e-3,
            learning_rate='constant',
            eta0=1e-5,
            shuffle=True,
        )
        self.initialized = False
        self.init_weights()
    
    def init_weights(self):
        self.model.coef_ = np.random.normal(0, 0.1, self.n_tilings*len(self.bounds))
        self.model.intercept_ = 0

    def rescale(self, state):
        return (state - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def tile_encode(self, state):
        z = np.zeros((self.n_tilings, len(state)))
        state = self.rescale(state)

        for tiling in range(self.n_tilings):
            s = (state + self.shifts*tiling) % 1
            s = (s * self.n_tiles)
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                print("Input contains NaN or Inf values")
                print("State", state)
            z[tiling,:] = s.astype(int)

        return z
    
    def fit(self, X, y):
        Z = np.array([self.tile_encode(x) for x in X])
        Z = Z.reshape(Z.shape[0], -1)
        if self.initialized:
            self.model.partial_fit(Z, y)
        else:
            self.model.fit(Z, y)
            self.initialized = True

    def update(self, X, y):
        Z = np.array([self.tile_encode(x) for x in X])
        Z = Z.reshape(Z.shape[0], -1)
        if self.initialized:
            self.model.partial_fit(Z, y)
        else:
            self.model.fit(Z, y)
            self.initialized = True
        mse = mean_squared_error(y, self.model.predict(Z))
        return mse
    
    def predict(self, state):
        z = self.tile_encode(state).reshape(1, -1)
        try:
            return self.model.predict(z)[0]
        except NotFittedError:
            return 0



from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    

    f = lambda x: np.random.normal(0, 0.5) + np.sin(x[0]*x[1]) + np.log(x[0] + x[1] + 1) * np.sqrt(x[0] + x[1] + 1)    
    X = np.random.rand(100, 2) * 3
    y = np.array([f(x) for x in X])

    bounds = np.array([[0, 3], [0, 3]])
    n_tiles = 8
    n_tilings = 32
    shifts = np.array([0.1, 0.3])

    tiling = TilingApproximatorMedium(bounds, n_tiles, n_tilings, shifts)

    # Fit the model
    tiling.fit(X, y)

    # Encode the first data point
    z = tiling.tile_encode(X[0])
    print("Encoded state:", z)

    # Predict the first data point
    y_preds = [tiling.predict(x) for x in X]
    train_mse = mean_squared_error(y, y_preds)
    print("Train MSE:", train_mse)

    X_test = np.random.rand(100, 2) * 3
    y_test = np.array([f(x) for x in X_test])
    y_preds = [tiling.predict(x) for x in X_test]
    test_mse = mean_squared_error(y_test, y_preds)
    print("Test MSE:", test_mse)

    # Show the weights
    coefs = tiling.model.coef_
    print("Weights mean:", np.mean(coefs), "std:", np.std(coefs))
    print(1/len(coefs))





    
    
    
