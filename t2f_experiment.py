# CELL 1: Import thư viện và thiết lập dữ liệu

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Thiết lập thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

def generate_synthetic_time_series(n_samples=2000):
    """Tạo chuỗi thời gian giả lập có xu hướng, mùa vụ và nhiễu."""
    np.random.seed(42)
    time = np.arange(n_samples)
    trend = 0.05 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 1.5, n_samples)
    return trend + seasonality + noise

series = generate_synthetic_time_series(n_samples=1500)
window_size = 30
print(f"Đã tạo thành công chuỗi dữ liệu với {len(series)} mẫu.")



# CELL 2: Môi trường Học tăng cường (RL Environment)

class TimeSeriesEnv:
    def __init__(self, data, window_size=30, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.current_step = window_size

    def reset(self):
        self.current_step = self.window_size
        return self._get_state()

    def _get_state(self):
        return self.data[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        target = self.data[self.current_step]
        mse = (action - target) ** 2
        
        # Phần thưởng kết hợp sai số và độ chính xác hướng xu hướng (Directional Accuracy)
        prev_value = self.data[self.current_step - 1]
        actual_direction = np.sign(target - prev_value)
        pred_direction = np.sign(action - prev_value)
        direction_reward = 1.0 if actual_direction == pred_direction else -1.0
        
        reward = -float(mse) + 0.5 * direction_reward
        self.current_step += 1
        done = self.current_step >= len(self.data) - self.horizon
        next_state = self._get_state() if not done else np.zeros(self.window_size)
        return next_state, reward, done, target



# CELL 3: Kiến trúc Mạng Neural (Actor, Critic & Supervised LSTM)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state_features = state.unsqueeze(-1)
        out, (hn, cn) = self.lstm(state_features)
        action = self.fc(hn[-1])
        return action.squeeze(-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state_features = state.unsqueeze(-1)
        out, (hn, cn) = self.lstm(state_features)
        value = self.fc(hn[-1])
        return value.squeeze(-1)

class SupervisedLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SupervisedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_feat = x.unsqueeze(-1)
        out, (hn, cn) = self.lstm(x_feat)
        return self.fc(hn[-1]).squeeze(-1)



# CELL 4: Huấn luyện các mô hình

print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH T2F (ACTOR-CRITIC) ===")
env = TimeSeriesEnv(data=series, window_size=window_size)
actor = ActorNetwork(window_size, hidden_dim=64).to(device)
critic = CriticNetwork(window_size, hidden_dim=64).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

n_episodes = 20
for episode in range(n_episodes):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        action = actor(state_tensor)
        next_state, reward, done, target = env.step(action.item())
        
        next_state_tensor = torch.FloatTensor(next_state).to(device)
        reward_tensor = torch.FloatTensor([reward]).to(device)

        state_val = critic(state_tensor)
        next_state_val = critic(next_state_tensor) if not done else torch.FloatTensor([0.0]).to(device)
        advantage = reward_tensor + (0.99 * next_state_val) - state_val

        critic_loss = advantage.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -actor(state_tensor) * advantage.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
print("Huấn luyện T2f hoàn tất!\n")

print("=== BẮT ĐẦU HUẤN LUYỆN SUPERVISED LSTM (BASELINE) ===")
sup_model = SupervisedLSTM(hidden_dim=64).to(device)
sup_optimizer = optim.Adam(sup_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_sup, y_sup = [], []
for i in range(window_size, len(series)):
    X_sup.append(series[i-window_size:i])
    y_sup.append(series[i])
X_sup = torch.FloatTensor(np.array(X_sup)).to(device)
y_sup = torch.FloatTensor(np.array(y_sup)).to(device)

sup_model.train()
for epoch in range(30):
    sup_optimizer.zero_grad()
    outputs = sup_model(X_sup)
    loss = criterion(outputs, y_sup)
    loss.backward()
    sup_optimizer.step()
print("Huấn luyện Supervised LSTM hoàn tất!")



# CELL 5: Đánh giá so sánh và Trực quan hóa

test_env = TimeSeriesEnv(data=series, window_size=window_size)
state = test_env.reset()
done = False

t2f_preds, sup_preds, baseline_preds, actuals = [], [], [], []
actor.eval()
sup_model.eval()

with torch.no_grad():
    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        pred_t2f = actor(state_tensor).item()
        pred_sup = sup_model(state_tensor).item()
        pred_baseline = state[-1] # Naive baseline
        
        next_state, reward, done, target = test_env.step(pred_t2f)
        
        t2f_preds.append(pred_t2f)
        sup_preds.append(pred_sup)
        baseline_preds.append(pred_baseline)
        actuals.append(target)
        state = next_state

t2f_preds = np.array(t2f_preds)
sup_preds = np.array(sup_preds)
baseline_preds = np.array(baseline_preds)
actuals = np.array(actuals)

def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    dir_true = np.sign(y_true[1:] - y_true[:-1])
    dir_pred = np.sign(y_pred[1:] - y_pred[:-1])
    dir_acc = np.mean(dir_true == dir_pred) * 100
    return mse, mae, dir_acc

m_base = evaluate(actuals, baseline_preds)
m_sup = evaluate(actuals, sup_preds)
m_t2f = evaluate(actuals, t2f_preds)

df_results = pd.DataFrame({
    'Thuật toán': ['Naive Baseline', 'Supervised LSTM', 'T2f (Actor-Critic RL)'],
    'MSE': [m_base[0], m_sup[0], m_t2f[0]],
    'MAE': [m_base[1], m_sup[1], m_t2f[1]],
    'Độ chính xác xu hướng (%)': [m_base[2], m_sup[2], m_t2f[2]]
})

print("\n================ BẢNG SO SÁNH HIỆU NĂNG ================")
print(df_results.to_string(index=False))

plt.figure(figsize=(14, 6))
plt.plot(actuals[:150], label="Thực tế (Actual)", color="black", linewidth=2.5)
plt.plot(baseline_preds[:150], label="Naive Baseline", color="gray", linestyle=":")
plt.plot(sup_preds[:150], label="Supervised LSTM", color="blue", linestyle="--")
plt.plot(t2f_preds[:150], label="T2f (Actor-Critic)", color="orange", linewidth=2)
plt.title("So sánh trực quan giữa T2f và các phương pháp khác", fontsize=13)
plt.xlabel("Bước thời gian", fontsize=11)
plt.ylabel("Giá trị", fontsize=11)
plt.legend()
plt.grid(True)
plt.show()