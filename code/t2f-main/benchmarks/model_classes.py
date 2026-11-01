import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pmdarima import auto_arima
from arch import arch_model
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from typing import List, Tuple, Dict, Optional
import warnings
import traceback


############################################################################################
### AutoArima
############################################################################################

class AutoARIMA:
    def __init__(self, seasonal=True, m=12):
        self.model = None
        self.seasonal = seasonal
        self.m = m

    def fit(self, y):
        # Check for constant data
        if np.allclose(y, y[0]):
            warnings.warn("Constant data detected. Using simple mean model.")
            self.model = ConstantMeanModel(y.mean())
            return

        # Check for sufficient data
        if len(y) <= self.m:
            warnings.warn(f"Insufficient data for seasonal modeling (length: {len(y)}, seasonal period: {self.m}). Using non-seasonal model.")
            self.seasonal = False
            self.m = 1

        # Remove NaN and infinite values
        y = y[np.isfinite(y)]

        if len(y) == 0:
            raise ValueError("No valid data points after removing NaN and infinite values.")

        try:
            self.model = auto_arima(y, seasonal=self.seasonal, m=self.m, suppress_warnings=True, error_action="ignore")
        except Exception as e:
            warnings.warn(f"AUTOARIMA fitting failed: {str(e)}. Using simple mean model.")
            self.model = ConstantMeanModel(y.mean())

    def predict(self, steps):
        if isinstance(self.model, ConstantMeanModel):
            return np.full(steps, self.model.mean)
        return self.model.predict(n_periods=steps)

class ConstantMeanModel:
    def __init__(self, mean):
        self.mean = mean

    def predict(self, n_periods):
        return np.full(n_periods, self.mean)

class ConstantMeanModel:
    def __init__(self, mean):
        self.mean = mean

    def predict(self, n_periods):
        return np.full(n_periods, self.mean)
    
### GARCH 
class AdaptiveScaler:
    def __init__(self, target_range=(1, 100)):
        self.target_range = target_range
        self.scale_factor = None
        self.center = None
        
    def fit(self, data: np.ndarray):
        """Fit scaler to data distribution."""
        # Handle outliers
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # Calculate center and scale
        self.center = np.median(cleaned_data)
        data_range = np.ptp(cleaned_data)
        target_range = self.target_range[1] - self.target_range[0]
        
        if data_range > 0:
            self.scale_factor = target_range / data_range
        else:
            self.scale_factor = 1.0
            
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data to target range."""
        if self.scale_factor is None:
            raise ValueError("Scaler must be fitted first")
            
        centered = data - self.center
        scaled = centered * self.scale_factor
        return scaled + np.mean(self.target_range)
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse scaling."""
        centered = data - np.mean(self.target_range)
        unscaled = centered / self.scale_factor
        return unscaled + self.center
    
class GARCHModel:
    def __init__(self, max_p=5, max_q=5, information_criterion='aic'):
        self.max_p = max_p
        self.max_q = max_q
        self.ic = information_criterion
        self.model = None
        self.best_orders = None
        self.scaler = AdaptiveScaler()

    def select_orders(self, y: np.ndarray) -> Tuple[int, int]:
        """Select optimal GARCH orders using information criteria."""
        best_score = np.inf
        best_orders = (1, 1)
        
        for p in range(1, self.max_p + 1):
            for q in range(1, self.max_q + 1):
                try:
                    model = arch_model(y, vol='GARCH', p=p, q=q)
                    result = model.fit(disp='off')
                    
                    # Get information criterion score
                    if self.ic == 'aic':
                        score = result.aic
                    elif self.ic == 'bic':
                        score = result.bic
                    else:
                        score = result.hqic
                    
                    if score < best_score:
                        best_score = score
                        best_orders = (p, q)
                        
                except Exception:
                    continue
                    
        return best_orders

    def detect_volatility_clustering(self, y: np.ndarray) -> bool:
        """Check if series exhibits volatility clustering."""
        # Calculate returns
        returns = np.diff(y) / y[:-1]
        
        # Calculate squared returns
        squared_returns = returns ** 2
        
        # Check autocorrelation of squared returns
        acf = np.correlate(squared_returns, squared_returns, mode='full')
        acf = acf[len(acf)//2:]
        
        # Test significance
        confidence = 1.96 / np.sqrt(len(returns))
        return any(abs(acf[1:11]) > confidence)
    
    def fit(self, y: np.ndarray):
        """Fit GARCH model with optimal parameters."""
        # Check for volatility clustering
        if not self.detect_volatility_clustering(y):
            warnings.warn("Series may not exhibit GARCH effects")
            
        # Scale data
        self.scaler.fit(y)
        y_scaled = self.scaler.transform(y)
        
        # Select optimal orders
        self.best_orders = self.select_orders(y_scaled)
        
        try:
            self.model = arch_model(y_scaled, vol='GARCH', 
                                  p=self.best_orders[0], 
                                  q=self.best_orders[1])
            self.res = self.model.fit(disp='off')
        except Exception as e:
            raise ValueError(f"GARCH fitting failed: {str(e)}")
            
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions with confidence intervals."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
            
        forecast = self.res.forecast(horizon=steps)
        predictions = self.scaler.inverse_transform(
            forecast.mean.values[-1]
        )
        
        return predictions

############################################################################################
### N-BEATS
############################################################################################

class NBeatsNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x.reshape(-1, self.network[0].in_features))

class NBeatsModel:
    def __init__(self, input_size, output_size, hidden_size=256, epochs=50, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NBeatsNet(input_size, output_size, hidden_size).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.HuberLoss(delta=1.0)
        
        self.input_mean = None
        self.input_std = None
        self.target_mean = None
        self.target_std = None

    def preprocess(self, data, is_target=False, fit=False):
        data_np = data.cpu().numpy() if torch.is_tensor(data) else np.array(data)
        if fit:
            if is_target:
                self.target_mean = np.mean(data_np)
                self.target_std = np.std(data_np)
                if self.target_std == 0:
                    self.target_std = 1
            else:
                self.input_mean = np.mean(data_np)
                self.input_std = np.std(data_np)
                if self.input_std == 0:
                    self.input_std = 1
                    
        mean = self.target_mean if is_target else self.input_mean
        std = self.target_std if is_target else self.input_std
        return (data_np - mean) / std

    def postprocess(self, data):
        return data * self.target_std + self.target_mean

    def fit(self, X, y):
        X_scaled = self.preprocess(X, fit=True)
        y_scaled = self.preprocess(y, is_target=True, fit=True)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(X_scaled)), shuffle=True)
        
        best_loss = float('inf')
        patience = 5
        counter = 0
        
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
            
            if total_loss < best_loss:
                best_loss = total_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    def predict(self, X, target_len=None):
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.preprocess(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            pred_scaled = self.model(X_tensor).cpu().numpy()
            predictions = self.postprocess(pred_scaled)
            
            if target_len:
                predictions = predictions[:target_len]
            if len(predictions.shape) == 2 and predictions.shape[0] == 1:
                predictions = predictions.flatten()
            
            return predictions

    def evaluate(self, X_test, y_test):
        target_len = y_test.shape[1] if len(y_test.shape) > 1 else len(y_test)
        y_true = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
        predictions = self.predict(X_test, target_len=target_len)
        return predictions[:len(y_true)]

############################################################################################
### Prophet
############################################################################################

class SimpleForecaster():
    def __init__(self, target_col='value'):
        self.model = None
        self.target_col = target_col

    def fit(self, df):
        # Convert DataFrame to series if necessary
        if isinstance(df, pd.DataFrame):
            series = df[self.target_col]
        else:
            series = df
        
        # Fit Simple Exponential Smoothing model
        self.model = SimpleExpSmoothing(series).fit()

    def predict(self, steps):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.forecast(steps)
        
class ProphetModel():
    def __init__(self, timestamp_col='value', target_col='value'):
        self.model = None
        self.fallback_model = None
        self.timestamp_col = timestamp_col
        self.target_col = target_col

    def fit(self, df):
        # Prophet requires columns 'ds' and 'y'
        df = df.rename(columns={self.timestamp_col: 'ds', self.target_col: 'y'})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = Prophet()
                self.model.fit(df)
        except Exception as e:
            print(f"Prophet fitting failed: {str(e)}. Using ExponentialSmoothing as fallback.")
            self.fallback_model = ExponentialSmoothing(df['y'].values, seasonal_periods=7, trend='add', seasonal='add').fit()

    def predict(self, future_dates):
        if self.model is not None:
            future = pd.DataFrame({'ds': future_dates})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.model.predict(future)
            return forecast['yhat'].values
        elif self.fallback_model is not None:
            return self.fallback_model.forecast(len(future_dates))
        else:
            raise ValueError("Model has not been fitted yet.")

############################################################################################        
### GRU
############################################################################################

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Changed to output size 1

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class GRUForecaster:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, epochs=50, batch_size=32):
        self.model = GRUModel(input_size, hidden_size, num_layers, output_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.output_size = output_size

    def create_sequences(self, data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length, :-1]  # Exclude the target variable
            target = data[i+seq_length, -1]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    def fit(self, X, y):
        # Scale X and y separately
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Combine scaled X and y
        combined_data = np.column_stack((X_scaled, y_scaled))
        
        # Create sequences
        seq_length = min(10, len(X) - 1)
        X_seq, y_seq = self.create_sequences(combined_data, seq_length)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)  # Add dimension to match output

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)  # Now dimensions match
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss:.4f}')

    def predict(self, X, steps):
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        predictions = []

        input_seq = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])
        for _ in range(steps):
            # Convert to tensor
            X_tensor = torch.FloatTensor(input_seq).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                pred_scaled = self.model(X_tensor).cpu().numpy()
                predictions.append(pred_scaled[0, 0])  # Take the single prediction

            # Update input sequence for next prediction
            if len(predictions) < steps:
                # Update the input sequence by sliding window
                input_seq = np.roll(input_seq, -1, axis=1)
                input_seq[0, -1, :] = X_scaled[-1, :]  # Use last known features

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions)
        return predictions.flatten()[:steps]

############################################################################################
### TCN
############################################################################################

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Ensure x is in the correct format (batch, channels, sequence length)
        if len(x.shape) == 3 and x.shape[1] != self.tcn.network[0].conv1.in_channels:
            x = x.transpose(1, 2)  # Transpose if needed
        output = self.tcn(x)
        output = output[:, :, -1]  # Take the last time step
        output = self.linear(output)
        return output

# class TCNForecaster:
#     def __init__(self, input_size, output_size, num_channels=None, kernel_size=2, dropout=0.2, epochs=100, batch_size=32):
#         self.input_size = input_size
#         self.output_size = output_size
#         self.kernel_size = kernel_size
#         self.dropout = dropout
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.model = None
#         self.scaler_X = MinMaxScaler()
#         self.scaler_y = MinMaxScaler()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.min_sequence_length = 2  # Minimum required sequence length

#     def _create_model(self, sequence_length):
#         """Dynamically create TCN model based on sequence length"""
#         # Calculate number of layers based on sequence length
#         num_layers = min(4, max(2, int(np.log2(sequence_length))))
        
#         # Create channel sizes that decrease by half each layer
#         base_channels = min(64, max(32, sequence_length * 2))
#         channels = [base_channels // (2**i) for i in range(num_layers)]
#         channels = [max(16, c) for c in channels]  # Ensure minimum channel size
        
#         return TCNModel(
#             input_size=self.input_size,
#             output_size=self.output_size,
#             num_channels=channels,
#             kernel_size=self.kernel_size,
#             dropout=self.dropout
#         ).to(self.device)

#     def create_sequences(self, data, seq_length):
#         """Create sequences with adaptive length"""
#         # Ensure minimum sequence length
#         if len(data) < self.min_sequence_length + self.output_size:
#             raise ValueError(f"Not enough data points. Need at least {self.min_sequence_length + self.output_size}")

#         # Use adaptive sequence length if needed
#         actual_seq_length = min(seq_length, len(data) - self.output_size)
#         actual_seq_length = max(self.min_sequence_length, actual_seq_length)

#         sequences = []
#         targets = []
        
#         for i in range(len(data) - actual_seq_length - self.output_size + 1):
#             seq = data[i:i+actual_seq_length, :self.input_size]
#             target = data[i+actual_seq_length:i+actual_seq_length+self.output_size, -1]
#             sequences.append(seq)
#             targets.append(target)

#         return np.array(sequences), np.array(targets), actual_seq_length

#     def fit(self, X, y):
#         """Fit model with adaptive architecture"""
#         # Scale inputs
#         X_scaled = self.scaler_X.fit_transform(X)
#         y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
#         combined_data = np.column_stack((X_scaled, y_scaled))

#         # Determine sequence length based on data
#         suggested_seq_length = min(int(len(X) * 0.2), self.output_size * 3)
#         suggested_seq_length = max(self.min_sequence_length, suggested_seq_length)

#         try:
#             X_seq, y_seq, actual_seq_length = self.create_sequences(
#                 combined_data, suggested_seq_length
#             )
#         except ValueError as e:
#             raise ValueError(f"Failed to create sequences: {str(e)}")

#         # Create model with appropriate architecture
#         self.model = self._create_model(actual_seq_length)

#         # Prepare data for training
#         X_tensor = torch.FloatTensor(X_seq).to(self.device)
#         y_tensor = torch.FloatTensor(y_seq).to(self.device)

#         # Create DataLoader with appropriate batch size
#         batch_size = min(self.batch_size, len(X_seq))
#         dataset = TensorDataset(X_tensor, y_tensor)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         # Training
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(self.model.parameters())
        
#         self.model.train()
#         for epoch in range(self.epochs):
#             total_loss = 0
#             for batch_X, batch_y in dataloader:
#                 optimizer.zero_grad()
                
#                 if batch_X.dim() == 3:
#                     batch_X = batch_X.transpose(1, 2)
                
#                 outputs = self.model(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
            
#             if (epoch + 1) % 10 == 0:
#                 print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}')

#     def predict(self, X, steps):
#         """Generate predictions with robust error handling and validation"""
#         if self.model is None:
#             raise ValueError("Model has not been fitted yet")

#         try:
#             # Input validation
#             if not isinstance(X, np.ndarray):
#                 X = np.array(X)
#             if np.any(np.isnan(X)) or np.any(np.isinf(X)):
#                 raise ValueError("Input contains NaN or infinite values")

#             # Scale input data
#             X_scaled = self.scaler_X.transform(X)
            
#             # Use as much history as available while respecting minimum length
#             history_length = max(self.min_sequence_length, min(len(X_scaled), self.output_size * 2))
#             X_seq = X_scaled[-history_length:].reshape(1, history_length, self.input_size)
            
#             # Convert to tensor with validation
#             X_tensor = torch.FloatTensor(X_seq).to(self.device)
#             if X_tensor.dim() == 3:
#                 X_tensor = X_tensor.transpose(1, 2)
            
#             # Generate predictions
#             self.model.eval()
#             with torch.no_grad():
#                 predictions_scaled = self.model(X_tensor).cpu().numpy()
                
#                 # Validate predictions
#                 if np.any(np.isnan(predictions_scaled)) or np.any(np.isinf(predictions_scaled)):
#                     print("Warning: Model generated invalid predictions, using fallback")
#                     # Use simple moving average as fallback
#                     fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
#                     return fallback_pred[:steps]
                
#             # Reshape and scale predictions with validation
#             predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
            
#             # Validate final predictions
#             if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
#                 print("Warning: Invalid values after inverse transform, using fallback")
#                 fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
#                 return fallback_pred[:steps]
            
#             # Ensure correct shape and length
#             predictions = predictions.flatten()
#             if len(predictions) < steps:
#                 predictions = np.pad(predictions, (0, steps - len(predictions)), mode='edge')
#             predictions = predictions[:steps]
            
#             # Final validation
#             if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
#                 print("Warning: Final predictions invalid, using fallback")
#                 fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
#                 return fallback_pred[:steps]
                
#             return predictions
            
#         except Exception as e:
#             print(f"Error in prediction: {str(e)}, using fallback")
#             # Fallback to simple average prediction
#             fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
#             return fallback_pred[:steps]

class TCNForecaster:
    def __init__(self, input_size, output_size, num_channels=None, kernel_size=2, dropout=0.2, epochs=100, batch_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_sequence_length = max(2, output_size)  # Use at least output_size as minimum

    def _create_model(self, sequence_length):
        """Dynamically create TCN model based on sequence length"""
        # Calculate number of layers based on sequence length
        num_layers = min(4, max(2, int(np.log2(sequence_length))))
        
        # Create channel sizes that decrease by half each layer
        base_channels = min(64, max(32, sequence_length * 2))
        channels = [base_channels // (2**i) for i in range(num_layers)]
        channels = [max(16, c) for c in channels]  # Ensure minimum channel size
        
        return TCNModel(
            input_size=self.input_size,
            output_size=self.output_size,
            num_channels=channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

    def create_sequences(self, data, seq_length):
        """Create sequences with adaptive length and fallback for small datasets"""
        # For very small datasets, use simpler approach
        if len(data) < self.min_sequence_length + self.output_size:
            print(f"Warning: Small dataset detected (length={len(data)}). Using simplified sequence creation.")
            # Use minimum possible sequence length
            actual_seq_length = max(2, len(data) - self.output_size)
            if actual_seq_length < 2:
                raise ValueError("Insufficient data points for sequence creation")
        else:
            # Use adaptive sequence length if we have enough data
            actual_seq_length = min(seq_length, len(data) - self.output_size)
            actual_seq_length = max(self.min_sequence_length, actual_seq_length)

        sequences = []
        targets = []
        
        for i in range(len(data) - actual_seq_length - self.output_size + 1):
            seq = data[i:i+actual_seq_length, :self.input_size]
            target = data[i+actual_seq_length:i+actual_seq_length+self.output_size, -1]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets), actual_seq_length

    def fit(self, X, y):
        """Fit model with adaptive architecture"""
        # Scale inputs
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        combined_data = np.column_stack((X_scaled, y_scaled))

        # Determine sequence length based on data
        suggested_seq_length = min(int(len(X) * 0.2), self.output_size * 3)
        suggested_seq_length = max(self.min_sequence_length, suggested_seq_length)

        try:
            X_seq, y_seq, actual_seq_length = self.create_sequences(
                combined_data, suggested_seq_length
            )
        except ValueError as e:
            raise ValueError(f"Failed to create sequences: {str(e)}")

        # Create model with appropriate architecture
        self.model = self._create_model(actual_seq_length)

        # Prepare data for training
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create DataLoader with appropriate batch size
        batch_size = min(self.batch_size, len(X_seq))
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                if batch_X.dim() == 3:
                    batch_X = batch_X.transpose(1, 2)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def predict(self, X, steps):
        """Generate predictions with robust error handling and validation"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        try:
            # Input validation
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Input contains NaN or infinite values")

            # Scale input data
            X_scaled = self.scaler_X.transform(X)
            
            # Use as much history as available while respecting minimum length
            history_length = max(self.min_sequence_length, min(len(X_scaled), self.output_size * 2))
            X_seq = X_scaled[-history_length:].reshape(1, history_length, self.input_size)
            
            # Convert to tensor with validation
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            if X_tensor.dim() == 3:
                X_tensor = X_tensor.transpose(1, 2)
            
            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                predictions_scaled = self.model(X_tensor).cpu().numpy()
                
                # Validate predictions
                if np.any(np.isnan(predictions_scaled)) or np.any(np.isinf(predictions_scaled)):
                    print("Warning: Model generated invalid predictions, using fallback")
                    # Use stored fallback value if available, otherwise use mean
                    fallback_value = self.fallback_prediction if self.fallback_prediction is not None else np.mean(X[:, 0])
                    return np.full(steps, fallback_value)
                
            # Reshape and scale predictions with validation
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
            
            # Validate final predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print("Warning: Invalid values after inverse transform, using fallback")
                fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
                return fallback_pred[:steps]
            
            # Ensure correct shape and length
            predictions = predictions.flatten()
            if len(predictions) < steps:
                predictions = np.pad(predictions, (0, steps - len(predictions)), mode='edge')
            predictions = predictions[:steps]
            
            # Final validation
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print("Warning: Final predictions invalid, using fallback")
                fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
                return fallback_pred[:steps]
                
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}, using fallback")
            # Fallback to simple average prediction
            fallback_pred = np.mean(X[:, 0]) * np.ones(steps)
            return fallback_pred[:steps]


############################################################################################
## DeepAR local
############################################################################################

# class LocalDeepARDataset(Dataset):
#     def __init__(self, data: pd.DataFrame, covariates: pd.DataFrame, window_size: int, 
#                  pred_len: int):
#         self.window_size = window_size
#         self.pred_len = pred_len
        
#         # Prepare sequences for single series
#         self.sequences = []
#         self.covariate_sequences = []
#         self.targets = []
        
#         series_data = data.values.reshape(-1)  # Ensure 1D array
#         covariate_data = covariates.values
        
#         for i in range(len(series_data) - window_size - pred_len + 1):
#             self.sequences.append(series_data[i:i+window_size])
#             self.covariate_sequences.append(covariate_data[i:i+window_size])
#             self.targets.append(series_data[i+window_size:i+window_size+pred_len])

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)
#         covariates = torch.FloatTensor(self.covariate_sequences[idx])
#         target = torch.FloatTensor(self.targets[idx])
        
#         return sequence, covariates, target

class LocalDeepARDataset(Dataset):
    def __init__(self, data: pd.DataFrame, covariates: pd.DataFrame, window_size: int, 
                 pred_len: int):
        self.window_size = window_size
        self.pred_len = pred_len
        
        # Prepare sequences for single series
        self.sequences = []
        self.covariate_sequences = []
        self.targets = []
        
        series_data = data.values.reshape(-1)  # Ensure 1D array
        covariate_data = covariates.values
        
        # Check if we have enough data for at least one sequence
        if len(series_data) < window_size + pred_len:
            # Handle short series by padding
            if len(series_data) > 0:
                # Pad the series to minimum required length
                pad_length = window_size + pred_len - len(series_data)
                series_data = np.pad(series_data, (0, pad_length), mode='edge')
                covariate_data = np.pad(covariate_data, ((0, pad_length), (0, 0)), mode='edge')
            else:
                raise ValueError("Series is empty")

        # Create sequences
        for i in range(len(series_data) - window_size - pred_len + 1):
            self.sequences.append(series_data[i:i+window_size])
            self.covariate_sequences.append(covariate_data[i:i+window_size])
            self.targets.append(series_data[i+window_size:i+window_size+pred_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)
        covariates = torch.FloatTensor(self.covariate_sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        
        return sequence, covariates, target

class LocalDeepARModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LocalDeepARModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor, covariates: torch.Tensor, 
               hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        batch_size = x.size(0)
        
        # Combine target variable with covariates
        combined_input = torch.cat([x, covariates], dim=2)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        output, hidden = self.lstm(combined_input, hidden)
        
        mu = self.mu_layer(output)
        sigma = F.softplus(self.sigma_layer(output))
        
        return mu, sigma, hidden
    
class LocalDeepARPredictor:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 input_length: int, 
                 output_length: int, 
                 learning_rate: float = 0.001, 
                 epochs: int = 100, 
                 batch_size: int = 32,
                 patience: int = 100,      # For early stopping
                 dropout:float = 0.2,
                 target_col:str = 'value'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.input_length = input_length
        self.output_length = output_length
        self.dropout = dropout
        self.target_col = target_col
        
        # Dictionary to store individual models and scalers for each series
        self.models = {}
        self.scalers = {}
        
    def gaussian_likelihood(self, mu: torch.Tensor, sigma: torch.Tensor, 
                          target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(-1)
        return 0.5 * torch.log(2 * np.pi * sigma.pow(2)) + \
               0.5 * (target - mu).pow(2) / sigma.pow(2)
    
    # def fit_series(self, series_id: str, X: pd.DataFrame, y: pd.DataFrame, 
    #               covariates: pd.DataFrame):
    #     # Create new model and scaler for this series
    #     self.models[series_id] = LocalDeepARModel(
    #         self.input_size, self.hidden_size, self.num_layers, self.dropout
    #     ).to(self.device)
    #     self.scalers[series_id] = StandardScaler()
        
    #     # Scale the target variable for this series
    #     y_scaled = pd.DataFrame(
    #         self.scalers[series_id].fit_transform(y.values.reshape(-1, 1)),
    #         index=y.index,
    #         # columns=y.columns,
    #         columns = [self.target_col]
    #     )

    #     # Adapt sequence lengths to available data
    #     effective_local_length = min(self.input_length, max(2, len(y_scaled) // 3))
    #     effective_global_length = min(self.output_length, max(2, len(y_scaled) // 3))

    #     print(f"Series {series_id}: Using output_length={effective_global_length} " 
    #                f"(original={self.output_length})")

    #     # Create dataset for this series
    #     dataset = LocalDeepARDataset(
    #         y_scaled, 
    #         covariates, 
    #         effective_local_length,  # Use adapted length 
    #         effective_global_length   # Use adapted length
    #     )

    #     dataloader = DataLoader(
    #         dataset, 
    #         batch_size=min(self.batch_size, max(2, len(dataset))), 
    #         shuffle=True
    #     )
        
    #     # Initialize optimizer for this model
    #     optimizer = torch.optim.Adam(self.models[series_id].parameters(), 
    #                                lr=self.learning_rate)
        
    #     best_loss = float('inf')
    #     patience_counter = 0
    #     best_model_state = None
        
    #     # Training loop with early stopping
    #     self.models[series_id].train()
    #     for epoch in range(self.epochs):
    #         total_loss = 0
    #         num_batches = 0
            
    #         for batch_x, batch_cov, batch_y in dataloader:
    #             batch_x = batch_x.to(self.device, non_blocking=True)
    #             batch_cov = batch_cov.to(self.device, non_blocking=True)
    #             batch_y = batch_y.to(self.device, non_blocking=True)
                
    #             optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                
    #             mu, sigma, _ = self.models[series_id](batch_x, batch_cov)
    #             loss = self.gaussian_likelihood(mu[:, -1:], sigma[:, -1:], batch_y).mean()
                
    #             loss.backward()
    #             optimizer.step()
                
    #             total_loss += loss.item()
    #             num_batches += 1
            
    #         avg_loss = total_loss / num_batches
            
    #         # Early stopping check
    #         if avg_loss < best_loss:
    #             best_loss = avg_loss
    #             patience_counter = 0
    #             best_model_state = self.models[series_id].state_dict()
    #         else:
    #             patience_counter += 1
            
    #         if epoch % 10 == 0:
    #             print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
            
    #         # Early stopping
    #         if patience_counter >= self.patience:
    #             print(f"Early stopping at epoch {epoch+1}")
    #             break
        
    #     # Restore best model
    #     if best_model_state is not None:
    #         self.models[series_id].load_state_dict(best_model_state)

    def fit_series(self, series_id: str, X: pd.DataFrame, y: pd.DataFrame, 
              covariates: pd.DataFrame):
        # Create new model and scaler for this series
        self.models[series_id] = LocalDeepARModel(
            self.input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)
        self.scalers[series_id] = StandardScaler()
        
        # Scale the target variable for this series
        y_scaled = pd.DataFrame(
            self.scalers[series_id].fit_transform(y.values.reshape(-1, 1)),
            index=y.index,
            columns=[self.target_col]
        )

        # Calculate minimum required length
        min_required_length = self.input_length + self.output_length

        if len(y_scaled) < min_required_length:
            # print(f"Local DeepAR Warning: Series {series_id} is too short ({len(y_scaled)} points). "
            #     f"Minimum required length is {min_required_length}. Using shortened window.")
            
            # Adjust lengths based on available data
            effective_local_length = max(2, len(y_scaled) // 2)
            effective_global_length = max(1, len(y_scaled) // 4)
        else:
            effective_local_length = min(self.input_length, max(2, len(y_scaled) // 3))
            effective_global_length = min(self.output_length, max(1, len(y_scaled) // 3))

        print(f"Local DeepAR fit series {series_id}: Using input_length={effective_local_length}, "
            f"output_length={effective_global_length}")

        try:
            # Create dataset for this series
            dataset = LocalDeepARDataset(
                y_scaled, 
                covariates, 
                effective_local_length,
                effective_global_length
            )

            if len(dataset) == 0:
                print(f"Warning: No valid sequences could be created for series {series_id}")
                return

            # Create dataloader with at least 1 sample per batch
            batch_size = min(self.batch_size, max(1, len(dataset)))
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=True
            )

            # Initialize optimizer for this model
            optimizer = torch.optim.Adam(self.models[series_id].parameters(), 
                                    lr=self.learning_rate)
            
            best_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Training loop with early stopping
            self.models[series_id].train()
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = 0
                
                for batch_x, batch_cov, batch_y in dataloader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_cov = batch_cov.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                    
                    mu, sigma, _ = self.models[series_id](batch_x, batch_cov)
                    loss = self.gaussian_likelihood(mu[:, -1:], sigma[:, -1:], batch_y).mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = self.models[series_id].state_dict()
                else:
                    patience_counter += 1
                
                # if epoch % 10 == 0:
                #     print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
                
                # Early stopping
                if patience_counter >= self.patience:
                    print(f"Local DeepAR early stopping at epoch {epoch+1}")
                    break
            
            # Restore best model
            if best_model_state is not None:
                self.models[series_id].load_state_dict(best_model_state)

        except Exception as e:
            print(f"Local DeepAR error training series {series_id}: {str(e)}")
            return
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, covariates: pd.DataFrame):
        # Fit model separately for each series
        for series_id in X.index.get_level_values(0).unique():
            # print(f"Training model for series {series_id}")
            series_X = X.loc[series_id]
            series_y = y.loc[series_id]
            series_covariates = covariates.loc[series_id]
            self.fit_series(series_id, series_X, series_y, series_covariates)
    
    def predict(self, X: pd.DataFrame, covariates: pd.DataFrame, steps: int,
               num_samples: int = 100) -> Dict[str, np.ndarray]:
        predictions = {}
        
        for series_id in X.index.get_level_values(0).unique():
            if series_id not in self.models:
                print(f"No Local DeepAR model found for series {series_id}, skipping...")
                continue
                
            model = self.models[series_id]
            scaler = self.scalers[series_id]
            model.eval()
            
            with torch.no_grad():
                try:
                    # Initialize storage for multi-step predictions
                    series_predictions = np.zeros((num_samples, steps))
                    
                    # Get initial input sequence
                    series_data = X.loc[series_id].values
                    x = series_data[:, 0].reshape(-1, 1)  # Take first feature as target
                    x_scaled = scaler.transform(x)
                    cov = covariates.loc[series_id].values
                    
                    # Initial tensors
                    x_tensor = torch.FloatTensor(x_scaled).reshape(1, len(series_data), 1).to(self.device)
                    cov_tensor = torch.FloatTensor(cov).reshape(1, len(cov), -1).to(self.device)
                    
                    # Generate samples for each future step
                    for sample in range(num_samples):
                        current_x = x_tensor.clone()
                        current_cov = cov_tensor.clone()
                        
                        # Generate predictions step by step
                        for step in range(steps):
                            mu, sigma, _ = model(current_x, current_cov)
                            pred = torch.normal(mu[:, -1], sigma[:, -1])
                            series_predictions[sample, step] = pred.cpu().numpy()
                            
                            if step < steps - 1:
                                current_x = torch.cat([
                                    current_x[:, 1:, :],
                                    pred.reshape(1, 1, 1)
                                ], dim=1)
                                
                                current_cov = torch.cat([
                                    current_cov[:, 1:, :],
                                    current_cov[:, -1:, :].clone()
                                ], dim=1)
                    
                    # Calculate mean prediction across samples
                    mean_predictions = np.mean(series_predictions, axis=0)
                    final_predictions = scaler.inverse_transform(
                        mean_predictions.reshape(-1, 1)
                    ).flatten()
                    
                    predictions[series_id] = final_predictions
                    
                except Exception as e:
                    print(f"Error processing series {series_id}: {str(e)}")
                    raise
        
        return predictions
    
############################################################################################    
### DeepAR global
############################################################################################

class GlobalDeepARDataset(Dataset):
    def __init__(self, data: pd.DataFrame, covariates: pd.DataFrame, window_size: int, 
                 pred_len: int, series_ids: List[str]):
        self.window_size = window_size
        self.pred_len = pred_len
        
        # Prepare sequences for each series
        self.sequences = []
        self.covariate_sequences = []
        self.targets = []
        
        for series_id in series_ids:
            series_data = data.loc[series_id].values.reshape(-1)  # Ensure 1D array
            covariate_data = covariates.loc[series_id].values
            
            for i in range(len(series_data) - window_size - pred_len + 1):
                self.sequences.append(series_data[i:i+window_size])
                self.covariate_sequences.append(covariate_data[i:i+window_size])
                self.targets.append(series_data[i+window_size:i+window_size+pred_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)  # Add feature dimension
        covariates = torch.FloatTensor(self.covariate_sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        
        return sequence, covariates, target

class GlobalDeepARModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(GlobalDeepARModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layer for processing historical values and covariates
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Output layers for distribution parameters
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor, covariates: torch.Tensor, 
               hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Ensure x has the right shape (batch, seq_len, 1)
        if x.dim() == 4:
            x = x.squeeze(3)
        
        # Ensure covariates has the right shape (batch, seq_len, input_size)
        if covariates.dim() == 2:
            covariates = covariates.unsqueeze(0)
        
        # Combine target variable with covariates
        combined_input = torch.cat([x, covariates], dim=2)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # Process through LSTM
        output, hidden = self.lstm(combined_input, hidden)
        
        # Generate distribution parameters
        mu = self.mu_layer(output)
        sigma = F.softplus(self.sigma_layer(output))
        
        return mu, sigma, hidden
    
class GlobalDeepARPredictor:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 input_length: int, 
                 output_length: int,
                 learning_rate: float = 0.001,
                 epochs: int = 100, 
                 batch_size: int = 32,
                 dropout: float = 0.2,
                 target_col: str = 'value'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GlobalDeepARModel(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_length = input_length
        self.output_length = output_length
        self.scaler = StandardScaler()
        self.input_size = input_size
        self.dropout = dropout
        self.target_col = target_col
        
    def gaussian_likelihood(self, mu: torch.Tensor, sigma: torch.Tensor, 
                          target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(-1)
        return 0.5 * torch.log(2 * np.pi * sigma.pow(2)) + \
               0.5 * (target - mu).pow(2) / sigma.pow(2)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, covariates: pd.DataFrame):
        # Scale the target variable
        self.scaler.fit(y.values.reshape(-1, 1))
        y_scaled = pd.DataFrame(
            self.scaler.transform(y.values.reshape(-1, 1)),
            index=y.index,
            # columns=y.columns
            columns = [self.target_col]
        )
        
        # Create dataset
        series_ids = X.index.get_level_values(0).unique()

        dataset = GlobalDeepARDataset(
            y_scaled, 
            covariates, 
            self.input_length, 
            self.output_length,
            series_ids
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        print(f'Global DeepAR fit: {self.epochs} epochs')
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_cov, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_cov = batch_cov.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                mu, sigma, _ = self.model(batch_x, batch_cov)
                loss = self.gaussian_likelihood(mu[:, -1:], sigma[:, -1:], batch_y).mean()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Global DeepAR epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def predict(self, X: pd.DataFrame, covariates: pd.DataFrame, steps: int,
               num_samples: int = 100) -> Dict[str, np.ndarray]:
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            for series_id in X.index.get_level_values(0).unique():
                try:
                    # Initialize storage for multi-step predictions
                    series_predictions = np.zeros((num_samples, steps))
                    
                    # Get initial input sequence
                    series_data = X.loc[series_id].values
                    x = series_data[:, 0].reshape(-1, 1)  # Take first feature as target
                    x_scaled = self.scaler.transform(x)
                    cov = covariates.loc[series_id].values
                    
                    # Initial tensors
                    x_tensor = torch.FloatTensor(x_scaled).reshape(1, len(series_data), 1).to(self.device)
                    cov_tensor = torch.FloatTensor(cov).reshape(1, len(cov), -1).to(self.device)
                    
                    # Generate samples for each future step
                    for sample in range(num_samples):
                        current_x = x_tensor.clone()
                        current_cov = cov_tensor.clone()
                        
                        # Generate predictions step by step
                        for step in range(steps):
                            # Get prediction for current step
                            mu, sigma, _ = self.model(current_x, current_cov)
                            pred = torch.normal(mu[:, -1], sigma[:, -1])
                            
                            # Store prediction
                            series_predictions[sample, step] = pred.cpu().numpy()
                            
                            # Update input sequence for next step
                            if step < steps - 1:  # Don't need to update for last step
                                # Shift input sequence and add new prediction
                                current_x = torch.cat([
                                    current_x[:, 1:, :],
                                    pred.reshape(1, 1, 1)
                                ], dim=1)
                                
                                # Shift covariate sequence and repeat last covariate
                                current_cov = torch.cat([
                                    current_cov[:, 1:, :],
                                    current_cov[:, -1:, :].clone()
                                ], dim=1)
                    
                    # Calculate mean prediction across samples
                    mean_predictions = np.mean(series_predictions, axis=0)
                    
                    # Inverse transform predictions
                    final_predictions = self.scaler.inverse_transform(
                        mean_predictions.reshape(-1, 1)
                    ).flatten()
                    
                    predictions[series_id] = final_predictions
                    
                except Exception as e:
                    print(f"Error processing series {series_id}: {str(e)}")
                    raise
        
        return predictions