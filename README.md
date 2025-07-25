# UniSPP: A Universal and Interpretable Method for Enhancing Stock Price Prediction

**UniSPP (Universal Symbolic Pairwise Predictor)** is an interpretable framework for stock price forecasting that explicitly models pairwise feature interactions using symbolic operations and stochastic subgraph sampling. The system enhances predictive accuracy, learning efficiency, and interpretability by integrating learned interactions into classical and deep learning models.

[Final Report (PDF)](https://github.com/rohan-303/UniSPP/blob/main/Final_Report.pdf)  

---

## Overview

Stock price prediction is inherently challenging due to volatility and complex dependencies. UniSPP addresses this by:

- Modeling **symbolic feature interactions** (e.g., RSI × MACD)
- Using **stochastic subgraph sampling** to reduce the combinatorial cost of interaction space
- Feeding both raw and learned composite features into predictive models like **LSTM, GRU, MLP, Ridge**, and **XGBoost**

---

## Dataset & Preprocessing

- **Source**: AAPL (Apple Inc.) OHLCV stock data for the past **5 years**
- **Features**: 160 technical indicators:
  - Trend: SMA, EMA, TRIMA, TEMA
  - Momentum: RSI, ROC, Momentum
  - Volatility: ATR, Bollinger Bands
  - Volume-based: OBV, Chaikin Oscillator
- **Preprocessing**:
  - MinMax normalization
  - Missing value removal
  - Chronological split (80% train, 20% test)

---

## Methodology

### Symbolic Feature Interaction Graph
- Fully connected graph where:
  - Nodes = technical indicators
  - Edges = operations: `+`, `-`, `*`, or ∅ (no interaction)

### Stochastic Subgraph Sampling
- Efficiently samples feature pairs + operations
- Builds composite symbolic features (e.g., RSI × MACD)
- Concatenates them with raw features during training

### Predictive Models
- **GRU, LSTM, RNN, MLP** (PyTorch-based)
- **Ridge Regression** (linear baseline)
- **XGBoost** (ensemble decision trees)

---

## Training & Hyperparameter Optimization

- **Optimizer**: Adam
- **Loss Function**: MAE, RMSE, NRMSE
- **Batch Size**: 32–128
- **Epochs**: 100–250
- **Tuning**: Conducted using Optuna across:
  - Hidden size
  - Layers
  - Learning rate
  - Activation functions
  - Subgraph sampling size

---

## Results Summary

### Evaluation Metrics

| Model     | Setup           | MAE     | RMSE    | NRMSE   |
|-----------|------------------|---------|---------|---------|
| GRU       | RAW              | 0.0164  | 0.0224  | 0.0911  |
| GRU       | UniSPP           | **0.0126**  | **0.0204**  | **0.0828** |
| MLP       | UniSPP           | 0.0127  | 0.0206  | 0.0834  |
| RNN       | UniSPP           | 0.0126  | 0.0206  | 0.0832  |
| Ridge     | UniSPP           | 0.0136  | 0.0214  | 0.0861  |
| XGBoost   | UniSPP           | 0.0128  | 0.0203  | 0.0825  |

- UniSPP **outperformed both raw and manually engineered feature configurations**
- Models using UniSPP showed **faster convergence and lower training loss**
- Visualizations (heatmaps, attention weights) revealed **interpretable interactions** aligned with financial heuristics

---

## Key Takeaways

- **Higher accuracy**: Up to 20–25% reduction in MAE compared to RAW input
- **Better interpretability**: Highlights meaningful financial feature pairs
- **Faster convergence**: Reduced training time and more stable optimization

---

## Future Directions

- Extend to **multi-stock and portfolio prediction**
- Integrate with **Transformer and attention-based models**
- Use **reinforcement learning** for dynamic subgraph selection
- Apply **causal inference** for filtering spurious interactions
