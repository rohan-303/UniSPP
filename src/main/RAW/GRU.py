import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

output_dir = "/home/careinfolab/Dr_Luo/Rohan/UniSPP/Results/RAW/GRU"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("/home/careinfolab/Dr_Luo/Rohan/UniSPP/Dataset/AAPL_dataset.csv", index_col=0, parse_dates=True)
df.dropna(inplace=True)

features = df.columns.tolist()
features.remove("Target")
X = df[features].values
y = df["Target"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train_tensor = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_fn_name):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }.get(act_fn_name, nn.Identity())

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.act(out[:, -1, :])
        return self.fc(out)

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    act_fn = trial.suggest_categorical("act_fn", ["relu", "tanh", "sigmoid"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 50, 200)

    model = GRUModel(X_train_tensor.shape[2], hidden_dim, 1, num_layers, act_fn).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(num_epochs):
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy().flatten()
        rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best = study.best_params
with open(os.path.join(output_dir, "best_params.pkl"), "wb") as f:
    pickle.dump(best, f)

final_model = GRUModel(X_train_tensor.shape[2],
                       best["hidden_dim"],
                       1,
                       best["num_layers"],
                       best["act_fn"]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best["lr"])

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=best["batch_size"], shuffle=True)

epoch_losses = []
final_model.train()
for _ in range(best["num_epochs"]):
    total_loss = 0.0
    for xb, yb in loader:
        preds = final_model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_losses.append(total_loss)

final_model.eval()
with torch.no_grad():
    final_preds = final_model(X_test_tensor).cpu().numpy().flatten()
    y_true = y_test

mae = mean_absolute_error(y_true, final_preds)
rmse = np.sqrt(mean_squared_error(y_true, final_preds))
nrmse = rmse / (y_true.max() - y_true.min())

with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
    pickle.dump({"MAE": mae, "RMSE": rmse, "NRMSE": nrmse, "params": best}, f)

plt.figure()
plt.plot(epoch_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
plt.close()

print("Best Parameters:", best)
print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}")
