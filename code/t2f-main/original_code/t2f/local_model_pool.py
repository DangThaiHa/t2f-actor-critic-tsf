import numpy as np
import torch
import torch.nn as nn

class AdaptiveSequentialModel(nn.Module):
    """Sequential model that can handle variable input lengths"""
    def __init__(self, input_size, hidden_size, output_size, output_length, activation=nn.ReLU()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_length = output_length
        self.activation = activation
        
        # Input adaptation
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Global context through adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output generation
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_length * output_size),
        )
        
    def forward(self, x):
        # Ensure input is 2D: [sequence_length, features]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.squeeze(0)
            
        # Process each timestep
        features = self.input_layer(x)  # [seq_len, hidden]
        
        # Get global context through pooling
        features = features.transpose(0, 1).unsqueeze(0)  # [1, hidden, seq_len]
        global_features = self.global_pool(features).squeeze(-1)  # [1, hidden]
        
        # Generate output
        output = self.output_layer(global_features)
        output = output.view(-1, self.output_length, self.output_size)
        
        return output.squeeze(0)
    
class SimpleVAR:
    def __init__(self):
        self.coefficients = None
        self.lag = None

    def fit(self, X):
        """Fit VAR model with any length of data"""
        if len(X) == 1:
            self.coefficients = np.eye(X.shape[1])  # Identity for single point
            self.lag = 1
            return
            
        self.lag = min(len(X) - 1, max(1, len(X) // 4))
        X_lagged = np.column_stack([X[i:-(self.lag-i) or None] for i in range(self.lag)])
        y = X[self.lag:]
        
        # Use pseudo-inverse for better numerical stability
        self.coefficients = np.linalg.pinv(X_lagged) @ y

    def predict(self, X, steps):
        if self.coefficients is None:
            return np.zeros((steps, X.shape[1]))
            
        # For single point, repeat the same value
        if len(X) == 1:
            return np.tile(X, (steps, 1))
            
        predictions = []
        current = X[-self.lag:].reshape(-1)
        
        for _ in range(steps):
            next_step = current @ self.coefficients
            predictions.append(next_step)
            current = np.roll(current, -len(next_step))
            current[-len(next_step):] = next_step
            
        return np.array(predictions)

class WrappedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_length):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.output_length = output_length

    def forward(self, x):
        # Handle single timestep
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2 and x.shape[0] == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        output, _ = self.lstm(x)
        
        # For single point, repeat the output
        if output.shape[1] == 1:
            output = output.repeat(1, self.output_length, 1)
        else:
            output = output[:, -self.output_length:, :]
            
        predictions = self.linear(output)
        return predictions.squeeze(0)

class WrappedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_length):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.output_length = output_length

    def forward(self, x):
        # Ensure input is 3D: [batch_size, sequence_length, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # GRU expects input of shape (batch, seq_len, input_size)
        output, _ = self.gru(x)
        
        # Use the last 'output_length' time steps
        last_steps = output[:, -self.output_length:, :]
        predictions = self.linear(last_steps)
        
        return predictions.squeeze(0)  # Remove batch dimension if single sample
