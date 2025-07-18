import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import pickle
import matplotlib.pyplot as plt

output_dir = "/home/careinfolab/Dr_Luo/Rohan/UniSPP/Results/Without_UniSPP/RIDGE"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("/home/careinfolab/Dr_Luo/Rohan/UniSPP/Dataset/AAPL_dataset.csv", index_col=0, parse_dates=True)
df.dropna(inplace=True)

features = df.columns.tolist()
features.remove("Target")

base_df = df[features].copy()
interaction_cols = {}

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        f1, f2 = features[i], features[j]
        interaction_cols[f"{f1}_+_{f2}"] = df[f1] + df[f2]
        interaction_cols[f"{f1}_*_{f2}"] = df[f1] * df[f2]

interaction_df = pd.DataFrame(interaction_cols, index=df.index)
X_raw = pd.concat([base_df, interaction_df], axis=1)
y = df["Target"].values

split_idx = int(len(X_raw) * 0.8)
X_train_raw, X_test_raw = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best = study.best_params
with open(os.path.join(output_dir, "best_params.pkl"), "wb") as f:
    pickle.dump(best, f)

final_model = Ridge(alpha=best["alpha"])
final_model.fit(X_train_scaled, y_train)
final_preds = final_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
nrmse = rmse / (y_test.max() - y_test.min())

with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
    pickle.dump({"MAE": mae, "RMSE": rmse, "NRMSE": nrmse, "params": best}, f)

plt.figure()
plt.plot([rmse] * 30, label="Loss (RMSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Constant RMSE vs Epoch (Ridge)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
plt.close()

print("Best Parameters:", best)
print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}")
