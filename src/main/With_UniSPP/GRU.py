import numpy as np
import pandas as pd
import itertools
import optuna
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

output_dir = "/home/careinfolab/Dr_Luo/Rohan/UniSPP/Results/With_UniSPP/GRU"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("/home/careinfolab/Dr_Luo/Rohan/UniSPP/Dataset/AAPL_dataset.csv", index_col=0, parse_dates=True)
df.dropna(inplace=True)

INTERACTION_OPERATORS = {
    '+': lambda x, y: x + y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / (y + 1e-6),
    '∅': lambda x, y: None
}
OP_NAMES = list(INTERACTION_OPERATORS.keys())
features = df.columns.tolist()
features.remove("Target")
feature_index = {f: i for i, f in enumerate(features)}
edges = list(itertools.combinations(features, 2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

def sample_subgraph(edges, phi, feature_index, num_edges):
    subgraph = []
    selected_edges = np.random.choice(len(edges), num_edges, replace=False)
    for idx in selected_edges:
        f1, f2 = edges[idx]
        i, j = feature_index[f1], feature_index[f2]
        probs = phi[i, j]
        op_idx = np.random.choice(len(OP_NAMES), p=probs)
        subgraph.append((f1, f2, OP_NAMES[op_idx], (i, j, op_idx)))
    return subgraph

def apply_interactions(df, subgraph):
    new_features = pd.DataFrame(index=df.index)
    for f1, f2, op, _ in subgraph:
        if op == '∅': continue
        name = f"{f1}_{op}_{f2}"
        new_features[name] = INTERACTION_OPERATORS[op](df[f1], df[f2])
    return new_features

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_fn_name):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
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

def evaluate_gru(subgraph, df_train, df_test, base_features, gru_params):
    inter_train = apply_interactions(df_train, subgraph)
    inter_test = apply_interactions(df_test, subgraph)

    train_full = pd.concat([df_train[base_features], inter_train], axis=1)
    test_full = pd.concat([df_test[base_features], inter_test], axis=1)

    train_full['Target'] = df_train['Target']
    test_full['Target'] = df_test['Target']

    train_full.dropna(inplace=True)
    test_full.dropna(inplace=True)

    X_train = MinMaxScaler().fit_transform(train_full.drop(columns='Target'))
    y_train = train_full['Target'].values

    X_test = MinMaxScaler().fit_transform(test_full.drop(columns='Target'))
    y_test = test_full['Target'].values

    y_min, y_max = y_test.min(), y_test.max()

    X_train_tensor = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    model = GRUModel(input_dim=X_train_tensor.shape[2],
                     hidden_dim=gru_params["hidden_dim"],
                     output_dim=1,
                     num_layers=gru_params["num_layers"],
                     act_fn_name=gru_params["act_fn"]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=gru_params["lr"])

    epoch_losses = []
    for epoch in range(gru_params["epochs"]):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy().flatten()

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    nrmse = rmse / (y_max - y_min)

    plt.figure()
    plt.plot(epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
    plt.close()

    return mae, rmse, nrmse, epoch_losses

def unispp_optimize_with_phi(df_train, df_test, edges, phi, feature_index, base_features, gru_params, trials, samples_per_trial, num_edges, lr):
    best_metrics = (float('inf'), None, None, None, None)
    for _ in range(trials):
        for _ in range(samples_per_trial):
            subgraph = sample_subgraph(edges, phi, feature_index, num_edges)
            mae, rmse, nrmse, losses = evaluate_gru(subgraph, df_train, df_test, base_features, gru_params)
            if mae < best_metrics[0]:
                best_metrics = (mae, rmse, nrmse, subgraph, losses)
    return best_metrics

def objective(trial):
    gru_params = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "act_fn": trial.suggest_categorical("act_fn", ["relu", "tanh", "sigmoid"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 50, 200)
    }

    num_edges = trial.suggest_int("num_edges", 3, 10)
    lr_phi = trial.suggest_float("lr_phi", 0.001, 0.2)
    samples_per_trial = trial.suggest_int("samples_per_trial", 2, 6)

    phi = np.ones((len(features), len(features), len(OP_NAMES)))
    phi /= phi.sum(axis=-1, keepdims=True)

    mae, rmse, nrmse, subgraph, _ = unispp_optimize_with_phi(
        df_train, df_test, edges, phi, feature_index, features, gru_params,
        trials=5, samples_per_trial=samples_per_trial,
        num_edges=num_edges, lr=lr_phi
    )

    trial.set_user_attr("mae", mae)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("nrmse", nrmse)
    trial.set_user_attr("best_subgraph", subgraph)
    return mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_trial = study.best_trial

with open(os.path.join(output_dir, "best_subgraph.pkl"), "wb") as f:
    pickle.dump(best_trial.user_attrs["best_subgraph"], f)

with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
    pickle.dump({
        "MAE": best_trial.user_attrs["mae"],
        "RMSE": best_trial.user_attrs["rmse"],
        "NRMSE": best_trial.user_attrs["nrmse"],
        "params": best_trial.params
    }, f)

print("Best MAE:", best_trial.user_attrs["mae"])
print("Best RMSE:", best_trial.user_attrs["rmse"])
print("Best NRMSE:", best_trial.user_attrs["nrmse"])
print("Best Parameters:", best_trial.params)
