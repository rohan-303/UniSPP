import os
import pandas as pd
import numpy as np
import optuna
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

output_dir = "/home/careinfolab/Dr_Luo/Rohan/UniSPP/Results/RAW/XGBoost"
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
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "objective": "reg:squarederror",
        "verbosity": 0
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best = study.best_params
with open(os.path.join(output_dir, "best_params.pkl"), "wb") as f:
    pickle.dump(best, f)

final_model = XGBRegressor(**best)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)

mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
nrmse = rmse / (y_test.max() - y_test.min())

with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
    pickle.dump({"MAE": mae, "RMSE": rmse, "NRMSE": nrmse, "params": best}, f)

print("Best Parameters:", best)
print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}")
