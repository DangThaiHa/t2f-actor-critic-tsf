import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List, Tuple, Dict

from sklearn.preprocessing import StandardScaler

import os
import pickle

import random

from helpers import base_logger, smape, wsmae, pre_input, prepare_data, infer_frequency
# from local_model_pool import SimpleVAR, WrappedLSTM, WrappedGRU

# Custom exception for T2f-related errors
class T2fError(Exception):
    pass

# Helper class for Sequential models
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

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

class DilatedCNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation_base=2):
        super(DilatedCNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size, dilation=dilation_base, padding=(kernel_size-1)*dilation_base//2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size, dilation=dilation_base**2, padding=(kernel_size-1)*(dilation_base**2)//2)
        self.conv3 = nn.Conv1d(64, output_dim, kernel_size, dilation=dilation_base**3, padding=(kernel_size-1)*(dilation_base**3)//2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

class FEDFormerInspiredLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_basis=10):
        super(FEDFormerInspiredLayer, self).__init__()
        self.num_basis = num_basis
        self.linear = nn.Linear(num_basis, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        x_fft = torch.fft.fft(x, dim=-1).abs()
        _, top_k_indices = torch.topk(x_fft, k=min(self.num_basis, x_fft.size(-1)), dim=-1)
        top_k_values = torch.gather(x_fft, -1, top_k_indices)
        return self.linear(top_k_values)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, logger):
        super(Actor, self).__init__()
        self.encoder = nn.Linear(state_dim, 128)  # Adjusted input dimension
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.max_action = max_action
        self.logger = logger

    def forward(self, state):
        self.logger.debug(f"Actor input state shape: {state.shape}")
        x = F.relu(self.encoder(state))
        self.logger.debug(f"After encoder shape: {x.shape}")
        x = F.relu(self.fc1(x))
        self.logger.debug(f"After fc1 shape: {x.shape}")
        x = self.fc2(x)
        if torch.all(torch.abs(x) < 1e-3):
            return F.softmax(x, dim=-1)
        self.logger.debug(f"Actor output shape: {x.shape}")
        return self.max_action * torch.tanh(x)
    
    def double_q_clip(self, x, q1, q2, clip_range=0.5):
        q_min = torch.min(q1, q2)
        q_max = torch.max(q1, q2)
        clipped_x = torch.clamp(x, q_min - clip_range, q_max + clip_range)
        return clipped_x

    def select_action(self, state, q1, q2):
        x = self.forward(state)
        clipped_x = self.double_q_clip(x, q1, q2)
        
        # Softmax fallback
        if torch.allclose(clipped_x, x):
            return F.softmax(x, dim=-1)
        else:
            return clipped_x
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simplified architecture to maintain batch dimension
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # Ensure state and action have correct shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
            
        # Concatenate along feature dimension
        x = torch.cat([state, action], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
# TD3 implementation
class TD3:
    def __init__(self, state_dim, action_dim, max_action, logger):
        self.actor = Actor(state_dim, action_dim, max_action, logger)
        self.actor_target = Actor(state_dim, action_dim, max_action, logger)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters())

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters())

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if replay_buffer.size < 2:
            return

        for it in range(iterations):
            # Sample a batch of transitions from replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(2)
            
            # Convert to tensors with explicit shapes
            state = torch.FloatTensor(state)  # Shape: (2, state_dim)
            action = torch.FloatTensor(action)  # Shape: (2, action_dim)
            next_state = torch.FloatTensor(next_state)  # Shape: (2, state_dim)
            
            # Take mean of reward if its dimension is larger than batch size
            if len(reward.shape) > 1 or reward.shape[0] > 2:
                reward = np.mean(reward.reshape(-1, reward.shape[-1]), axis=1)[:2]
            reward = torch.FloatTensor(reward).view(2, 1)  # Force shape (2, 1)
            
            if len(done.shape) > 1:
                done = done.reshape(-1)[:2]
            done = torch.FloatTensor(done).view(2, 1)  # Force shape (2, 1)

            # Select next action according to target policy
            noise = torch.FloatTensor(action[:2].shape).normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * discount * target_Q.detach()

            # Optimize critics
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)

            # Optimize Critic 1
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        # Ensure we don't sample more than available
        batch_size = min(batch_size, len(self.storage))
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []
        
        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        
        return (np.array(batch_states), 
                np.array(batch_actions), 
                np.array(batch_next_states), 
                np.array(batch_rewards), 
                np.array(batch_dones))

    @property
    def size(self):
        return len(self.storage)

class T2f:
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int,
                 local_length: int, global_length: int,
                 series_indices: List[str], 
                 freq: str = None,
                 logger = None,
                 batch_size: int = 100,
                 timestamp_col = 'timestamp',
                 series_id_col = 'series_id',
                 epochs: int = 10):
        
        self.freq = freq
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.local_length = local_length
        self.global_length = global_length
        self.series_indices = series_indices
        self.timestamp_col = timestamp_col
        self.series_id_col = series_id_col
        self.logger = logger
        self.batch_size = batch_size
        self.epochs = epochs

        self.local_models = [
                SimpleVAR(),
                
                WrappedLSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    output_size=self.output_size,
                    output_length=self.local_length
                ),
                
                WrappedGRU(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    output_size=self.output_size,
                    output_length=self.local_length
                ),
                
                AdaptiveSequentialModel(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    output_length=self.local_length,
                    activation=nn.ReLU()
                ),
                
                AdaptiveSequentialModel(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    output_length=self.local_length,
                    activation=nn.Tanh()
                )
            ]
        
        self.num_models = len(self.local_models)

        # State dimensions
        self.local_state_dim = self.num_models * local_length * output_size
        self.global_state_dim = self.local_state_dim + input_size
        self.action_dim = self.num_models
        
        # Initialize TD3 model
        self.global_model = TD3(
            state_dim=self.global_state_dim,
            action_dim=self.action_dim,
            max_action=1,
            logger=logger
        )
        
        self.scaler = StandardScaler()

    def temporal_attention(self, series_data: np.ndarray, decay_factor: float = 0.1) -> np.ndarray:
        """
        Compute temporal attention weights with proper shape handling.
        """
        seq_length = len(series_data)
        
        # Base temporal weights
        temporal_weights = np.exp(np.arange(seq_length) / seq_length)
        
        # Ensure weights are properly shaped for broadcasting
        if seq_length > 1:
            volatility = np.abs(np.diff(series_data, axis=0))
            volatility = np.pad(volatility, ((1, 0), (0, 0)), mode='edge')  # Pad first point
            
            # Calculate volatility weights per feature
            volatility_norm = np.std(volatility, axis=0, keepdims=True)
            volatility_norm = np.where(volatility_norm == 0, 1, volatility_norm)
            volatility_weights = 1 + decay_factor * (volatility - np.mean(volatility, axis=0, keepdims=True)) / volatility_norm
            
            # Average volatility weights across features
            volatility_weights = np.mean(volatility_weights, axis=1)
        else:
            volatility_weights = np.ones(seq_length)
        
        # Combine weights
        combined_weights = temporal_weights * volatility_weights
        
        # Normalize
        return combined_weights / combined_weights.sum()
    
    def edge_case_handler(self, X_series: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Handle edge cases in data preprocessing"""
        if len(X_series) == 0:
            raise ValueError("Empty series provided")
        
        if len(X_series) == 1:
            return np.tile(X_series, (3, 1))
        
        if np.all(np.isclose(X_series, X_series[0])):
            if is_training:
                noise = np.random.normal(0, 1e-4, X_series.shape)
                return X_series + noise
            return X_series
        
        return X_series

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle irregular timestamps and missing values using updated pandas methods"""
        try:
            # Sort by index
            df = df.sort_index()
            processed_dfs = []
            
            for series_id in df.index.get_level_values(self.series_id_col).unique():
                series_df = df.loc[series_id].copy()
                
                # Resample if frequency is specified
                if self.freq is not None:
                    if self.freq == 'S':
                        self.freq = 's' ## handle future warning
                    series_df = series_df.resample(self.freq).asfreq()
                
                # Handle missing values:
                # 1. Forward fill with limit
                series_df = series_df.ffill(limit=2)
                
                # 2. Backward fill with limit
                series_df = series_df.bfill(limit=2)
                
                # 3. Linear interpolate remaining gaps
                series_df = series_df.interpolate(method='linear', limit_direction='both')
                
                # Restore MultiIndex
                series_df.index = pd.MultiIndex.from_product(
                    [[series_id], series_df.index],
                    names=[self.series_id_col, self.timestamp_col]
                )
                
                processed_dfs.append(series_df)
            
            # Combine all processed series
            processed_df = pd.concat(processed_dfs)
            
            # Sort the final dataframe
            return processed_df.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise T2fError(f"Failed to preprocess data: {str(e)}")
        

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate data structure and quality"""
        try:
            assert isinstance(df.index, pd.MultiIndex), "DataFrame must have MultiIndex"
            assert all(name in df.index.names for name in [self.series_id_col, self.timestamp_col]), \
                   "Index must have series_id and timestamp"
            
            required_features = df.columns[:self.input_size]
            missing_ratio = df[required_features].isnull().mean()
            if (missing_ratio > 0.3).any():
                raise ValueError(f"Features {required_features[missing_ratio > 0.3]} have too many missing values")
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            raise T2fError(f"Data validation failed: {str(e)}")
        
    def local_stage(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp], np.ndarray]:
        try:
            local_preds = {}
            for (series_id, timestamp), group in X.groupby(level=[0, 1]):
                if len(group) == 0:
                    continue

                X_series = group.values
                y_series = y.loc[series_id].loc[:timestamp].values
                
                # Get attention weights
                attention_weights = self.temporal_attention(X_series)
                attention_weights = attention_weights.reshape(-1, 1)
                
                series_preds = []
                for i, model in enumerate(self.local_models):
                    try:
                        if isinstance(model, SimpleVAR):
                            weighted_X = X_series * attention_weights
                            model.fit(weighted_X)
                            pred = model.predict(weighted_X, steps=self.local_length)
                        else:
                            # For all neural network models
                            X_tensor = torch.FloatTensor(X_series)
                            attention_tensor = torch.FloatTensor(attention_weights)
                            weighted_X = X_tensor * attention_tensor
                            
                            with torch.no_grad():
                                pred = model(weighted_X)
                                pred = pred.detach().numpy()

                        pred = self.ensure_prediction_shape(pred)
                        series_preds.append(pred)
                        
                    except Exception as e:
                        self.logger.warning(f"Model {i} failed for series {series_id}: {str(e)}")
                        # Add zero predictions as fallback
                        fallback_pred = np.zeros((self.local_length, self.output_size))
                        series_preds.append(fallback_pred)

                if series_preds:
                    local_preds[(series_id, timestamp)] = np.array(series_preds)

            return local_preds
                
        except Exception as e:
            self.logger.error(f"Error in local stage: {str(e)}")
            raise T2fError(f"Failed to complete local stage: {str(e)}") from e

    def global_stage(self, local_preds: Dict[Tuple[str, pd.Timestamp], np.ndarray], 
                    X: pd.DataFrame) -> np.ndarray:
        try:
            # Initialize global_input with correct size
            global_input = np.zeros(self.global_state_dim)
            
            for i, series_id in enumerate(self.series_indices):
                series_data = X.loc[series_id]
                series_preds = []
                series_features = []
                
                # Get temporal attention weights
                attention_weights = self.temporal_attention(series_data.values)
                
                for timestamp in series_data.index:
                    if (series_id, timestamp) in local_preds:
                        preds = local_preds[(series_id, timestamp)]
                        # Ensure predictions have correct shape
                        if preds.size > 0:
                            series_preds.append(preds.reshape(-1))  # Flatten predictions
                            series_features.append(series_data.loc[timestamp].values)
                
                if series_preds:
                    # Ensure all predictions have the same shape
                    max_length = max(pred.shape[0] for pred in series_preds)
                    padded_preds = [
                        np.pad(pred, (0, max_length - len(pred)), 'constant')
                        for pred in series_preds
                    ]
                    
                    # Apply attention-weighted averaging
                    weights = attention_weights[-len(series_preds):]
                    weights = weights / weights.sum()  # Renormalize
                    
                    avg_preds = np.average(padded_preds, axis=0, weights=weights)
                    avg_features = np.average(series_features, axis=0, weights=weights)
                    
                    # Ensure shapes match expected dimensions
                    pred_dim = self.num_models * self.local_length * self.output_size
                    avg_preds = avg_preds[:pred_dim]  # Truncate if necessary
                    
                    # Combine predictions and features
                    combined = np.concatenate([
                        avg_preds,
                        avg_features[:self.input_size]
                    ])
                    
                    # Ensure combined array matches global_state_dim
                    if len(combined) > self.global_state_dim:
                        combined = combined[:self.global_state_dim]
                    elif len(combined) < self.global_state_dim:
                        combined = np.pad(combined, (0, self.global_state_dim - len(combined)))
                    
                    global_input = combined

            self.logger.debug(f"Global input shape: {global_input.shape}")
            global_input = global_input.reshape(1, -1)
            weights = self.global_model.select_action(global_input)
            self.logger.debug(f"Global weights shape: {weights.shape}")
            
            return weights.reshape(self.num_models)
            
        except Exception as e:
            self.logger.error(f"Error in global stage: {str(e)}")
            raise T2fError(f"Failed to complete global stage: {str(e)}") from e


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, covariates: pd.DataFrame):
        try:
            # Preprocess and validate data
            X = self.preprocess_data(X)
            y = self.preprocess_data(y)
            covariates = self.preprocess_data(covariates)
            
            self.validate_data(X)
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            
            replay_buffer = ReplayBuffer(max_size=1000) ## can be adjusted
            
            for epoch in range(self.epochs):
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                
                # Get local predictions
                local_preds = self.local_stage(X_scaled, y)
                
                # Get global weights
                global_weights = self.global_stage(local_preds, X_scaled)
                
                # Initialize state and reward arrays with correct dimensions
                global_state = np.zeros(self.global_state_dim)
                global_reward = np.zeros(self.action_dim)
                
                for i, series_id in enumerate(self.series_indices):
                    series_state = []
                    series_reward = []
                    
                    series_data = X_scaled.loc[series_id]
                    attention_weights = self.temporal_attention(series_data.values)
                    
                    for timestamp in series_data.index:
                        if (series_id, timestamp) in local_preds:
                            preds = local_preds[(series_id, timestamp)]
                            
                            # Ensure state has correct dimensions
                            state = np.zeros(self.global_state_dim)
                            pred_flat = preds.flatten()[:self.local_state_dim]
                            features = series_data.loc[timestamp].values[:self.input_size]
                            
                            state[:len(pred_flat)] = pred_flat
                            state[self.local_state_dim:self.local_state_dim + len(features)] = features
                            
                            series_state.append(state)
                            
                            # Calculate rewards
                            y_true = y.loc[series_id].loc[timestamp:].values[:self.local_length]
                            if len(y_true) > 0:
                                preds_aligned = preds[:, :len(y_true)]
                                covariates_aligned = covariates.loc[series_id].loc[timestamp:].values[:len(y_true), -1]
                                
                                smape_scores = np.array([smape(y_true, pred) for pred in preds_aligned])
                                wsmae_scores = np.array([wsmae(y_true, pred, covariates_aligned) for pred in preds_aligned])
                                reward = -0.5 * smape_scores - 0.5 * wsmae_scores
                                series_reward.append(reward)
                    
                    if series_state:
                        # Apply attention-weighted averaging
                        weights = attention_weights[-len(series_state):]
                        weights = weights / weights.sum()
                        
                        global_state = np.average(series_state, axis=0, weights=weights)
                        if series_reward:
                            global_reward = np.average(series_reward, axis=0, weights=weights[-len(series_reward):])
                
                # Add to replay buffer
                replay_buffer.add((
                    global_state,
                    global_weights,
                    global_state,
                    global_reward,
                    False
                ))
                
                # Train TD3
                if replay_buffer.size >= 2:
                    self.global_model.train(
                        replay_buffer=replay_buffer,
                        iterations=10, ## can be adjusted
                        batch_size=min(self.batch_size, replay_buffer.size)
                    )
        
        except Exception as e:
            self.logger.error(f"Error in fit method: {str(e)}")
            raise T2fError(f"Failed to fit T2f model: {str(e)}") from e

    def fit_incremental(self, X: pd.DataFrame, y: pd.DataFrame, covariates: pd.DataFrame, 
                    #batch_size: int = 100, 
                   validation_split: float = 0.1):
        """
        Incremental fitting for large datasets with temporal attention.
        
        Args:
            X: Feature DataFrame with MultiIndex (series_id, timestamp)
            y: Target DataFrame with MultiIndex (series_id, timestamp)
            covariates: Covariate DataFrame with MultiIndex (series_id, timestamp)
            batch_size: Number of timestamps to process in each batch
            validation_split: Proportion of data to use for validation
        """
        try:
            self.logger.info("Starting incremental fitting...")
            batch_size=self.batch_size

            # Preprocess all data
            X = self.preprocess_data(X)
            y = self.preprocess_data(y)
            covariates = self.preprocess_data(covariates)
            
            self.validate_data(X)
            
            # Scale features
            if not hasattr(self, 'scaler_fitted'):
                X_scaled = self.scaler.fit_transform(X)
                self.scaler_fitted = True
            else:
                X_scaled = self.scaler.transform(X)
            
            X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            
            # Split into batches by time
            timestamps = sorted(X.index.get_level_values(self.timestamp_col).unique())
            n_batches = max(1, len(timestamps) // batch_size)
            
            # Split validation data
            if validation_split > 0:
                val_size = int(len(timestamps) * validation_split)
                train_timestamps = timestamps[:-val_size]
                val_timestamps = timestamps[-val_size:]
            else:
                train_timestamps = timestamps
                val_timestamps = []
            
            # Initialize replay buffer with increased size for batch processing
            replay_buffer = ReplayBuffer(max_size=max(1000, batch_size * 10))
            
            best_val_loss = float('inf')
            no_improvement_count = 0
            
            for epoch in range(self.epochs):
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                epoch_losses = []
                
                # Training phase
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min((batch_idx + 1) * batch_size, len(train_timestamps))
                    batch_timestamps = train_timestamps[batch_start:batch_end]
                    
                    # Get batch data
                    batch_mask = X_scaled.index.get_level_values(self.timestamp_col).isin(batch_timestamps)
                    X_batch = X_scaled[batch_mask]
                    y_batch = y[batch_mask]
                    covariates_batch = covariates[batch_mask]
                    
                    # Initialize batch state and reward
                    batch_state = np.zeros(self.state_dim)
                    batch_reward = np.zeros(self.action_dim)
                    
                    # Process batch with temporal attention
                    local_preds = self.local_stage(X_batch, y_batch)
                    global_weights = self.global_stage(local_preds, X_batch)
                    
                    # Process each series in batch
                    for i, series_id in enumerate(self.series_indices):
                        series_mask = X_batch.index.get_level_values(self.series_id_col) == series_id
                        series_data = X_batch[series_mask]
                        
                        if len(series_data) == 0:
                            continue
                        
                        # Compute temporal attention for series
                        attention_weights = self.temporal_attention(series_data.values)
                        
                        series_state = []
                        series_reward = []
                        
                        for timestamp in series_data.index.get_level_values(self.timestamp_col):
                            if (series_id, timestamp) in local_preds:
                                # Get predictions and prepare state
                                preds = local_preds[(series_id, timestamp)]
                                current_state = np.concatenate([
                                    preds.flatten(),
                                    series_data.loc[series_id].loc[timestamp].values,
                                    covariates_batch.loc[series_id].loc[timestamp].values
                                ])
                                series_state.append(current_state)
                                
                                # Calculate attention-weighted rewards
                                y_true = y_batch.loc[series_id].loc[timestamp:].values[:self.local_length]
                                local_preds_aligned = preds[:, :len(y_true)]
                                covariates_aligned = covariates_batch.loc[series_id].loc[timestamp:].values[:self.local_length, -1]
                                
                                cur_attention = attention_weights[len(attention_weights)-len(y_true):]
                                
                                # Calculate metrics with temporal attention
                                smape_scores = np.array([
                                    smape(y_true * cur_attention, pred * cur_attention) 
                                    for pred in local_preds_aligned
                                ])
                                wsmae_scores = np.array([
                                    wsmae(y_true * cur_attention, pred * cur_attention, covariates_aligned) 
                                    for pred in local_preds_aligned
                                ])
                                reward = -0.5 * smape_scores - 0.5 * wsmae_scores
                                series_reward.append(reward)
                        
                        if series_state:
                            # Apply temporal attention to state and reward aggregation
                            series_attention = attention_weights[-len(series_state):]
                            
                            start_idx = i * (self.num_models * self.local_length * self.output_size + self.input_size)
                            end_idx = start_idx + self.num_models * self.local_length * self.output_size + self.input_size
                            
                            batch_state[start_idx:end_idx] = np.average(series_state, axis=0, weights=series_attention)
                            batch_reward[i*self.num_models:(i+1)*self.num_models] = np.average(series_reward, axis=0, weights=series_attention)
                    
                    # Add batch to replay buffer
                    replay_buffer.add((
                        batch_state,
                        global_weights.flatten(),
                        batch_state,
                        batch_reward,
                        False
                    ))
                    
                    # Train on batch
                    if replay_buffer.size >= 2:
                        self.global_model.train(
                            replay_buffer=replay_buffer,
                            iterations=10,
                            batch_size=min(100, replay_buffer.size)
                        )
                    
                    epoch_losses.append(-np.mean(batch_reward))
                    self.logger.debug(f"Batch {batch_idx + 1}/{n_batches} completed. Loss: {-np.mean(batch_reward):.4f}")
                
                # Validation phase
                if validation_split > 0:
                    val_loss = self._validate_on_timestamps(X_scaled, y, covariates, val_timestamps)
                    self.logger.info(f"Epoch {epoch + 1} - Training Loss: {np.mean(epoch_losses):.4f}, Validation Loss: {val_loss:.4f}")
                    
                    # Early stopping logic
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
                    if no_improvement_count >= 3:  # Early stopping after 3 epochs without improvement
                        self.logger.info("Early stopping triggered")
                        break
                else:
                    self.logger.info(f"Epoch {epoch + 1} completed. Average Loss: {np.mean(epoch_losses):.4f}")
            
            self.logger.info("Incremental fitting completed.")
            
        except Exception as e:
            self.logger.error(f"Error in incremental fit: {str(e)}")
            raise T2fError(f"Failed to fit T2f model incrementally: {str(e)}") from e

    def _validate_on_timestamps(self, X_scaled: pd.DataFrame, y: pd.DataFrame, 
                            covariates: pd.DataFrame, val_timestamps: list) -> float:
        """Helper method for validation during incremental fitting."""
        try:
            val_mask = X_scaled.index.get_level_values(self.timestamp_col).isin(val_timestamps)
            X_val = X_scaled[val_mask]
            y_val = y[val_mask]
            covariates_val = covariates[val_mask]
            
            # Get predictions
            local_preds = self.local_stage(X_val, y_val)
            global_weights = self.global_stage(local_preds, X_val)
            
            # Calculate validation loss
            total_loss = 0
            count = 0
            
            for series_id in self.series_indices:
                series_mask = X_val.index.get_level_values(self.series_id_col) == series_id
                if not any(series_mask):
                    continue
                    
                series_data = X_val[series_mask]
                attention_weights = self.temporal_attention(series_data.values)
                
                for timestamp in series_data.index.get_level_values(self.timestamp_col):
                    if (series_id, timestamp) in local_preds:
                        preds = local_preds[(series_id, timestamp)]
                        y_true = y_val.loc[series_id].loc[timestamp:].values[:self.local_length]
                        weighted_preds = np.sum(preds * global_weights.reshape(-1, 1, 1), axis=0)
                        
                        # Apply temporal attention to validation metrics
                        cur_attention = attention_weights[len(attention_weights)-len(y_true):]
                        loss = np.mean((weighted_preds[:len(y_true)] - y_true) ** 2 * cur_attention)
                        
                        total_loss += loss
                        count += 1
            
            return total_loss / max(1, count)
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            return float('inf')

    def ensure_prediction_shape(self, pred: np.ndarray) -> np.ndarray:
        """Ensure predictions have correct shape with proper broadcasting."""
        pred = np.asarray(pred)
        
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        
        if pred.shape[0] != self.local_length:
            if pred.shape[0] > self.local_length:
                pred = pred[:self.local_length]
            else:
                pad_length = self.local_length - pred.shape[0]
                pred = np.pad(pred, ((0, pad_length), (0, 0)), mode='edge')
        
        if pred.shape[1] != self.output_size:
            if pred.shape[1] > self.output_size:
                pred = pred[:, :self.output_size]
            else:
                pad_width = self.output_size - pred.shape[1]
                pred = np.pad(pred, ((0, 0), (0, pad_width)), mode='constant')
        
        return pred
    
    def save(self, path):
        """
        Save the T2f model with temporal attention configuration to the specified path.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # Save global TD3 model components
        torch.save(self.global_model.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.global_model.actor_target.state_dict(), os.path.join(path, 'actor_target.pth'))
        torch.save(self.global_model.critic_1.state_dict(), os.path.join(path, 'critic_1.pth'))
        torch.save(self.global_model.critic_1_target.state_dict(), os.path.join(path, 'critic_1_target.pth'))
        torch.save(self.global_model.critic_2.state_dict(), os.path.join(path, 'critic_2.pth'))
        torch.save(self.global_model.critic_2_target.state_dict(), os.path.join(path, 'critic_2_target.pth'))

        # Save local models
        for i, model in enumerate(self.local_models):
            if isinstance(model, SimpleVAR):
                with open(os.path.join(path, f'local_model_{i}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            elif isinstance(model, (WrappedLSTM, WrappedGRU)):
                torch.save(model.state_dict(), os.path.join(path, f'local_model_{i}.pth'))
            elif isinstance(model, nn.Sequential):
                torch.save(model.state_dict(), os.path.join(path, f'local_model_{i}.pth'))

        # Save scaler
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save model configuration and attributes
        attributes = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'local_length': self.local_length,
            'global_length': self.global_length,
            'series_indices': self.series_indices,
            'freq': self.freq,
            'timestamp_col': self.timestamp_col,
            'series_id_col': self.series_id_col,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }
        with open(os.path.join(path, 'attributes.pkl'), 'wb') as f:
            pickle.dump(attributes, f)

    @classmethod
    def load(cls, path, logger=None):
        """
        Load a T2f model with temporal attention from the specified path.
        """
        try:
            # Load attributes
            with open(os.path.join(path, 'attributes.pkl'), 'rb') as f:
                attributes = pickle.load(f)
            
            # Create new instance with loaded attributes
            if logger == None:
                logger = base_logger()
            attributes['logger'] = logger
            t2f = cls(**attributes)

            # Load global TD3 model components
            t2f.global_model.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
            t2f.global_model.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target.pth')))
            t2f.global_model.critic_1.load_state_dict(torch.load(os.path.join(path, 'critic_1.pth')))
            t2f.global_model.critic_1_target.load_state_dict(torch.load(os.path.join(path, 'critic_1_target.pth')))
            t2f.global_model.critic_2.load_state_dict(torch.load(os.path.join(path, 'critic_2.pth')))
            t2f.global_model.critic_2_target.load_state_dict(torch.load(os.path.join(path, 'critic_2_target.pth')))

            # Load local models
            for i in range(len(t2f.local_models)):
                try:
                    if isinstance(t2f.local_models[i], SimpleVAR):
                        with open(os.path.join(path, f'local_model_{i}.pkl'), 'rb') as f:
                            t2f.local_models[i] = pickle.load(f)
                    elif isinstance(t2f.local_models[i], (WrappedLSTM, WrappedGRU)):
                        t2f.local_models[i].load_state_dict(torch.load(os.path.join(path, f'local_model_{i}.pth')))
                    elif isinstance(t2f.local_models[i], nn.Sequential):
                        t2f.local_models[i].load_state_dict(torch.load(os.path.join(path, f'local_model_{i}.pth')))
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to load local model {i}: {str(e)}. Reinitializing.")
                    # Model will be reinitialized with default weights

            # Load scaler
            with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                t2f.scaler = pickle.load(f)

            return t2f

        except Exception as e:
            if logger:
                logger.error(f"Error loading model: {str(e)}")
            raise T2fError(f"Failed to load T2f model: {str(e)}")

    @staticmethod
    def is_valid_save(path) -> bool:
        """
        Check if a saved model directory contains all necessary components.
        """
        required_files = [
            'actor.pth',
            'actor_target.pth',
            'critic_1.pth',
            'critic_1_target.pth',
            'critic_2.pth',
            'critic_2_target.pth',
            'attributes.pkl',
            'scaler.pkl'
        ]
        
        # Check for required global model files
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False
        
        # Check for attributes to determine number of local models
        try:
            with open(os.path.join(path, 'attributes.pkl'), 'rb') as f:
                attributes = pickle.load(f)
                
            # Check for local model files
            for i in range(5):  # Fixed number of local models in current implementation
                if not (os.path.exists(os.path.join(path, f'local_model_{i}.pkl')) or 
                    os.path.exists(os.path.join(path, f'local_model_{i}.pth'))):
                    return False
                    
            return True
            
        except Exception:
            return False
        
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with temporal attention for new data.
        Predictions start from the last available timestamp in input data.
        """
        try:
            # Preprocess and validate input data
            X = self.preprocess_data(X)
            self.validate_data(X)

            self.logger.info(f"Starting prediction with input shape: {X.shape}")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            
            predictions = {}
            for series_id, group in X_scaled.groupby(level=0):
                if len(group) == 0:
                    self.logger.warning(f"Empty data for series {series_id}")
                    continue

                X_series = group.values
                
                # Log the time range for debugging
                self.logger.debug(f"Series {series_id} time range: {group.index.get_level_values(self.timestamp_col)[0]} to {group.index.get_level_values(self.timestamp_col)[-1]}")
                
                # Get temporal attention weights
                attention_weights = self.temporal_attention(X_series)
                attention_weights = attention_weights.reshape(-1, 1)
                
                # Generate local predictions
                local_preds = []
                for i, model in enumerate(self.local_models):
                    try:
                        if isinstance(model, SimpleVAR):
                            weighted_X = X_series * attention_weights
                            if len(weighted_X) == 1:
                                pred = np.tile(weighted_X, (self.local_length, 1))
                            else:
                                model.fit(weighted_X)
                                pred = model.predict(weighted_X, steps=self.local_length)
                                
                        else:  # Neural network models
                            X_tensor = torch.FloatTensor(X_series)
                            attention_tensor = torch.FloatTensor(attention_weights)
                            weighted_X = X_tensor * attention_tensor
                            
                            with torch.no_grad():
                                pred = model(weighted_X)
                                pred = pred.detach().numpy()
                        
                        pred = self.ensure_prediction_shape(pred)
                        local_preds.append(pred)
                        
                    except Exception as e:
                        self.logger.warning(f"Model {i} failed for series {series_id}: {str(e)}")
                        fallback_pred = np.zeros((self.local_length, self.output_size))
                        local_preds.append(fallback_pred)
                
                if local_preds:
                    # Stack all model predictions
                    local_preds = np.array(local_preds)  # [num_models, output_length, output_size]
                    
                    # Prepare global input
                    global_input = np.zeros(self.global_state_dim)
                    pred_flat = local_preds.reshape(-1)[:self.local_state_dim]
                    features = X_series[-1][:self.input_size]
                    
                    global_input[:len(pred_flat)] = pred_flat
                    global_input[self.local_state_dim:self.local_state_dim + len(features)] = features
                    
                    # Get ensemble weights
                    global_input = global_input.reshape(1, -1)
                    weights = self.global_model.select_action(global_input)
                    weights = weights.reshape(-1, 1, 1)  # [num_models, 1, 1]
                    
                    # Combine predictions with weights
                    final_pred = np.sum(local_preds * weights, axis=0)  # [output_length, output_size]
                    
                    # Get the last timestamp and calculate step size
                    timestamps = group.index.get_level_values(self.timestamp_col)
                    last_timestamp = timestamps[-1]
                    
                    if len(timestamps) >= 2:
                        # Calculate median time difference for more robust frequency inference
                        time_diffs = pd.Series(timestamps).diff().dropna()
                        typical_diff = time_diffs.median()  # Use median instead of mode
                        
                        # Generate future timestamps starting immediately after last timestamp
                        future_timestamps = []
                        current_timestamp = last_timestamp
                        for i in range(self.local_length):
                            current_timestamp = current_timestamp + typical_diff
                            future_timestamps.append(current_timestamp)
                            
                            # Log prediction points for debugging
                            self.logger.debug(f"Prediction point {i+1} for series {series_id}: {current_timestamp}")
                    else:
                        # Fallback for single timestamp: use daily frequency
                        future_timestamps = [
                            last_timestamp + pd.Timedelta(days=i+1)
                            for i in range(self.local_length)
                        ]
                    
                    # Store predictions with proper timestamps
                    for i, timestamp in enumerate(future_timestamps):
                        predictions[(series_id, timestamp)] = final_pred[i]
            
            # Convert predictions to DataFrame
            if predictions:
                pred_rows = []
                for (sid, ts), pred in predictions.items():
                    pred_rows.append({
                        'series_id': sid,
                        'timestamp': ts,
                        'prediction': float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
                    })
                
                # Sort by timestamp within each series
                pred_df = pd.DataFrame(pred_rows)
                pred_df = pred_df.sort_values(['series_id', 'timestamp'])
                pred_df = pred_df.set_index(['series_id', 'timestamp'])
                
                # Log prediction summary
                self.logger.info(f"Generated predictions for {len(pred_df.index.unique(level=0))} series")
                self.logger.info(f"Prediction time range: {pred_df.index.get_level_values(1).min()} to {pred_df.index.get_level_values(1).max()}")
            else:
                pred_df = pd.DataFrame(columns=['series_id', 'timestamp', 'prediction'])
                pred_df = pred_df.set_index(['series_id', 'timestamp'])

            return pred_df

        except Exception as e:
            self.logger.error(f"Error in predict method: {str(e)}")
            raise T2fError(f"Failed to generate predictions: {str(e)}") from e

def data_load(
        input_data_path: str,
        series_id_col: str,
        timestamp_col: str,
        timestamp_format: str,
        covariate_cols: list[str],
        weight_col: str,
        target_col: str,
        test_size: float,
        freq: str = None,
        logger = None,
        id_frac: float = 1,
        len_frac: float = 1
        ):
    
    """
    Run T2f with temporal attention on input data.
    Add freq and logger as optional parameters.
    """

    if logger == None:
        logger = base_logger()

    try:
        # Load data
        filename, file_extension = os.path.splitext(input_data_path)
        if file_extension == '.parquet':
            data = pd.read_parquet(input_data_path)
        elif file_extension == '.csv':
            data = pd.read_csv(input_data_path)
        else:
            logger.error(f"Error in data loading: incompatible format")
            raise T2fError(f"Load csv or parquet data")
        
                
        # Prepare data with automatic frequency inference
        prepared_df = pre_input(
            df=data,
            series_id_col=series_id_col,
            timestamp_col=timestamp_col,
            timestamp_format=timestamp_format,
            covariate_cols=covariate_cols,
            weight_col=weight_col,
            target_col=target_col,
        )
                
        # df fraction if needed based on available mem
        series_ids = list(prepared_df.index.get_level_values(series_id_col).unique())
        print(f"Preload data length: {len(prepared_df)}")
        print(f"Preload ids: {len(series_ids)}")

        if id_frac < 1:
            prepared_df_ = pd.DataFrame()
            for _ in random.sample(series_ids, k=int(id_frac*len(series_ids))):
                id_data_ = prepared_df[prepared_df.index.get_level_values(series_id_col).isin([_])]
                prepared_df_ = pd.concat([prepared_df_, id_data_], ignore_index=False, axis=0)
        
            prepared_df = prepared_df_.copy()
            del prepared_df_
        
        if len_frac < 1:
            prepared_df_ = pd.DataFrame()
            for _ in series_ids:
                id_data_ = prepared_df[prepared_df.index.get_level_values(series_id_col).isin([_])]
                if int(len_frac*len(id_data_))>0:
                    prepared_df_ = pd.concat([prepared_df_, id_data_[:int(len_frac*len(id_data_))]], ignore_index=False, axis=0)

            prepared_df = prepared_df_.copy()
            del prepared_df_
        
        series_ids = list(prepared_df.index.get_level_values(series_id_col).unique())
        print(f"Loaded data length: {len(prepared_df)}")
        print(f"Loaded ids: {len(series_ids)}")        
    
        if freq == None:
            freq = infer_frequency(data[timestamp_col],
                                   timestamp_format=timestamp_format,
                                   logger=logger)
                    
        # Split data
        train_df, test_df = prepare_data(
            prepared_df, 
            test_size=test_size, 
            timestamp_col=timestamp_col,
            series_id_col=series_id_col
        )

        return train_df, test_df, prepared_df, freq

    except Exception as e:
        logger.error(f"Error in data load: {str(e)}")
        raise T2fError(f"Failed to load data: {str(e)}") from e
            
def t2f_fit(
        train_df,
        test_df,
        prepared_df,
        freq: str,
        series_id_col: str,
        timestamp_col: str,
        covariate_cols: list[str],
        weight_col: str,
        target_col: str,
        local_length: int,
        global_length: int,
        epochs: int,
        model_save_path: str = None,
        hidden_size: int = None, #computed
        num_layers: int = None, #computed
        batch_size: int = 100,
        logger = None
        ):
    
    """
    Run T2f with temporal attention on input data.
    Add freq and logger as optional parameters.
    """

    if logger == None:
        logger = base_logger()

    try:
        # Load data
        series_ids = list(prepared_df.index.get_level_values(series_id_col).unique())

        # Network architecture parameters
        base_size = len(covariate_cols) * 4
        
        # Scale with series complexity
        series_factor = min(2, np.log2(len(series_ids)) / 2)
        
        # Scale with temporal complexity (use median series length instead of fixed input length)
        median_length = prepared_df.groupby(series_id_col).size().median()
        temporal_factor = min(2, np.log2(median_length) / 4)
        
        if hidden_size == None:
            # Calculate hidden size
            hidden_size = int(base_size * series_factor * temporal_factor)
            hidden_size = max(64, min(hidden_size, 512))
            hidden_size = 2 ** int(np.log2(hidden_size) + 0.5)
        
        if num_layers == None:
            # Calculate number of layers
            base_layers = max(2, min(6, int(np.log2(median_length) / 2)))
            feature_factor = min(2, len(covariate_cols) / 4)
            num_layers = int(base_layers * feature_factor)
    
        # Initialize T2f with temporal attention
        t2f = T2f(
            input_size=len(covariate_cols),
            output_size=1,  # Fixed for single target
            hidden_size=hidden_size,
            num_layers=num_layers,
            local_length=local_length,
            global_length=global_length,
            series_indices=series_ids,
            freq=freq,
            timestamp_col=timestamp_col,
            series_id_col=series_id_col,
            epochs=epochs,
            batch_size=batch_size,
            logger=logger
        )

        # Prepare training data
        X_train = train_df[covariate_cols]
        y_train = train_df[target_col]
        covariates_train = train_df[covariate_cols + [weight_col]]

        # Fit model
        logger.info("Starting model training...")
        t2f.fit(X_train, y_train, covariates_train)

        # Save model
        if model_save_path:
            logger.info(f"Saving model to {model_save_path}")
            if T2f.is_valid_save(model_save_path):
                logger.warning(f"Existing model found at {model_save_path}, overwriting...")
            t2f.save(model_save_path)
        else:
            logger.info("Skipping local model save, no path provided")

        return t2f

    except Exception as e:
        logger.error(f"Error in run_t2f: {str(e)}")
        raise T2fError(f"Failed to run T2f: {str(e)}") from e