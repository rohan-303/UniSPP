import numpy as np
import pandas as pd
import itertools
import optuna
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

output_dir = "/home/careinfolab/Dr_Luo/Rohan/UniSPP/Results/With_UniSPP/RIDGE"
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
        if op == '∅':
            continue
        name = f"{f1}_{op}_{f2}"
        new_features[name] = INTERACTION_OPERATORS[op](df[f1], df[f2])
    return new_features

def evaluate_ridge(subgraph, df_train, df_test, base_features, alpha):
    inter_train = apply_interactions(df_train, subgraph)
    inter_test = apply_interactions(df_test, subgraph)

    train_full = pd.concat([df_train[base_features], inter_train], axis=1)
    test_full = pd.concat([df_test[base_features], inter_test], axis=1)

    train_full['Target'] = df_train['Target']
    test_full['Target'] = df_test['Target']

    train_full.dropna(inplace=True)
    test_full.dropna(inplace=True)

    X_train = train_full.drop(columns='Target')
    y_train = train_full['Target'].values

    X_test = test_full.drop(columns='Target')
    y_test = test_full['Target'].values

    y_min, y_max = y_test.min(), y_test.max()

    X_train_scaled = MinMaxScaler().fit_transform(X_train)
    X_test_scaled = MinMaxScaler().fit_transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    nrmse = rmse / (y_max - y_min)
    return mae, rmse, nrmse

def unispp_optimize_with_phi(df_train, df_test, edges, phi, feature_index, base_features, alpha, trials, samples_per_trial, num_edges, lr):
    best_metrics = (float('inf'), None, None, None)
    for _ in range(trials):
        for _ in range(samples_per_trial):
            subgraph = sample_subgraph(edges, phi, feature_index, num_edges)
            mae, rmse, nrmse = evaluate_ridge(subgraph, df_train, df_test, base_features, alpha)
            if mae < best_metrics[0]:
                best_metrics = (mae, rmse, nrmse, subgraph)

            baseline = 0.01
            for _, _, _, (i, j, op_idx) in subgraph:
                grad = (mae - baseline)
                phi[i, j, op_idx] -= lr * grad
                phi[i, j] = np.maximum(phi[i, j], 1e-6)
                phi[i, j] /= phi[i, j].sum()
    return best_metrics

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    num_edges = trial.suggest_int("num_edges", 3, 10)
    lr = trial.suggest_float("lr", 0.001, 0.2)
    samples_per_trial = trial.suggest_int("samples_per_trial", 2, 8)

    phi = np.ones((len(features), len(features), len(OP_NAMES)))
    phi /= phi.sum(axis=-1, keepdims=True)

    mae, rmse, nrmse, subgraph = unispp_optimize_with_phi(
        df_train=df_train,
        df_test=df_test,
        edges=edges,
        phi=phi,
        feature_index=feature_index,
        base_features=features,
        alpha=alpha,
        trials=5,
        samples_per_trial=samples_per_trial,
        num_edges=num_edges,
        lr=lr
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
