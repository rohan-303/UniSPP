import os
import pandas as pd
import numpy as np
import optuna
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

output_dir = "/home/careinfolab/Dr_Luo/Rohan/UniSPP/Results/RAW/RIDGE"
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

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best = study.best_params
with open(os.path.join(output_dir, "best_params.pkl"), "wb") as f:
    pickle.dump(best, f)

final_model = Ridge(alpha=best["alpha"])
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)

mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
nrmse = rmse / (y_test.max() - y_test.min())

with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
    pickle.dump({"MAE": mae, "RMSE": rmse, "NRMSE": nrmse, "params": best}, f)

print("Best Parameters:", best)
print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}")
