# ================================
# AI-Powered Stock Trading Bot: All Features, Clean Structure, Visual Output
# ================================
# Supports: LSTM, Sentiment, RSI, Moving Averages, Multi-Stock, Portfolio Opt, Visualization

# 1. Install dependencies (Colab/Jupyter: run this cell if any import fails)
try:
    import yfinance, pandas, numpy, tensorflow, transformers, scipy, matplotlib, sklearn
except ImportError:
    import sys
    !{sys.executable} -m pip install numpy pandas yfinance scikit-learn tensorflow transformers matplotlib scipy

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# === Indicators ===
def compute_rsi(data, window=14):
    data = np.array(data).flatten()
    diff = np.diff(data)
    up = diff.clip(min=0)
    down = -1 * diff.clip(max=0)
    avg_gain = pd.Series(up).rolling(window).mean()
    avg_loss = pd.Series(down).rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.full(window, np.nan), rsi.values[window:]])

def moving_average(data, window=20):
    data = np.array(data).flatten()
    return pd.Series(data).rolling(window=window).mean().values

# === LSTM Model ===
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === Data Preparation ===
def preprocess_data(data):
    scaler = StandardScaler()
    normed = scaler.fit_transform(data)
    return normed, scaler

def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# === Sentiment Analysis ===
def fetch_news_sentiment(headlines=None):
    if not headlines:
        headlines = [
            "Strong earnings reported by tech companies.",
            "Markets react to new government policy.",
        ]
    sentiment = pipeline('sentiment-analysis')
    scores = [sentiment(text)[0]['score'] * (1 if sentiment(text)[0]['label']=='POSITIVE' else -1) for text in headlines]
    return np.mean(scores) if len(scores) > 0 else 0.0

# === Advisor & Colored Summary ===
def formatted_decision(decision):
    html_decisions = {
        "STRONG BUY": "<span style='color:#0f0;font-weight:bold;'>🟢🟢 STRONG BUY</span>",
        "BUY": "<span style='color:lime;font-weight:bold;'>🟢 BUY</span>",
        "STRONG SELL": "<span style='color:#f33;font-weight:bold;'>🔴🔴 STRONG SELL</span>",
        "SELL": "<span style='color:orange;font-weight:bold;'>🔴 SELL</span>",
        "HOLD": "<span style='color:gold;font-weight:bold;'>🟡 HOLD</span>",
    }
    return html_decisions.get(decision, decision)

def advisor_decision(pred, cur, sentiment_idx, rsi, ma20):
    if pred > cur and sentiment_idx > 0.1 and rsi < 30 and cur > ma20:
        return "STRONG BUY"
    elif pred > cur and sentiment_idx > 0 and rsi < 50:
        return "BUY"
    elif pred < cur and sentiment_idx < 0 and rsi > 70 and cur < ma20:
        return "STRONG SELL"
    elif pred < cur and sentiment_idx < 0:
        return "SELL"
    else:
        return "HOLD"

# === Visualization ===
def show_prediction_plot(data, pred_price, cur_price, ticker, decision):
    closes = data['Close'].values.flatten()
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-60:], closes[-60:], label="Last 60 Days")
    plt.scatter(data.index[-1], pred_price, marker='*', c='red', s=250, label="Predicted")
    plt.scatter(data.index[-1], cur_price, marker='o', c='green', s=150, label="Current")
    plt.title(f"{ticker}: Prediction ➡ {decision}", fontsize=14, color='navy')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def clear_text_output(ticker, cur_price, pred_price, stock_rsi, stock_ma20, sentiment, decision):
    summary = f"""
### {ticker}
- **Current Price:** `{cur_price:.2f}`
- **Predicted Next Price:** `{pred_price:.2f}`
- **RSI:** `{stock_rsi:.2f}`
- **20-Day MA:** `{stock_ma20:.2f}`
- **Sentiment:** `{sentiment:.2f}`
- **Advisor Suggests:** {formatted_decision(decision)}
---
"""
    display(Markdown(summary))

# === Portfolio Optimization ===
def optimize_portfolio(returns_data):
    n_assets = returns_data.shape[1]
    def objective(w):
        pr = np.sum(returns_data.mean() * w)
        pv = np.sqrt(np.dot(w.T, np.dot(returns_data.cov(), w)))
        return -pr / (pv + 1e-8)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    bounds = tuple((0,1) for _ in range(n_assets))
    init_guess = np.repeat(1/n_assets, n_assets)
    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
    return result.x

# === Backtest ===
def backtest_lstm_bot(ticker="AAPL", period='6mo', seq_len=20, epochs=5):
    df = yf.download(ticker, period=period, interval='1d')
    if df.shape[0] <= seq_len: return None, None
    closes = df['Close'].values.flatten()
    ma20 = moving_average(closes)
    rsi = compute_rsi(closes)
    normed, scaler = preprocess_data(df[['Close']])
    X, y = prepare_sequences(normed, seq_len)
    model = build_lstm_model((seq_len, 1))
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    predictions = scaler.inverse_transform(model.predict(X)).flatten()
    real = df['Close'].values[seq_len:].flatten()
    test_rsi = rsi[seq_len:]
    test_ma20 = ma20[seq_len:]
    results = {"true": real, "pred": predictions, "rsi": test_rsi, "ma20": test_ma20, "dates": df.index[seq_len:]}
    return model, results

def plot_backtest(results, stockname="Stock"):
    plt.figure(figsize=(12,6))
    plt.plot(results["dates"], results["true"], label="Actual Close")
    plt.plot(results["dates"], results["pred"], label="LSTM Prediction")
    plt.title(f"{stockname} Backtest: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Main Bot ===
def stock_trading_bot():
    display(Markdown("<h2 style='color:#1464F4;font-weight:bold;'>✨ AI-Powered Stock Trading Bot ✨</h2>"))
    tickers = input("Enter comma-separated tickers (e.g., AAPL, TSLA, RELIANCE.NS): ").upper().replace(' ', '').split(',')
    period = '1y'
    seq_len = 20
    epochs = 8
    news = [
        "Markets rally as inflation slows.",
        "Tech company beats analyst expectations.",
        "Fed hints rate cut.",
        "Global supply issues."
    ]
    sentiment = fetch_news_sentiment(news)
    display(Markdown(f"<span style='color:purple;font-weight:bold;'>News Sentiment Index: <b>{sentiment:.2f}</b></span>"))
    portfolio = []
    for ticker in tickers:
        data = yf.download(ticker, period=period, interval='1d')
        if data.shape[0] <= seq_len+1:
            display(Markdown(f"<span style='color:#888;'>Not enough data for {ticker}</span>"))
            continue
        closes = data['Close'].values.flatten()
        ma20 = moving_average(closes)
        rsi = compute_rsi(closes)
        normed, scaler = preprocess_data(data[['Close']])
        X, y = prepare_sequences(normed, seq_len)
        model = build_lstm_model((seq_len, 1))
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        last_seq = normed[-seq_len:].reshape(1, seq_len, 1)
        pred_norm = model.predict(last_seq)[0,0]
        pred_price = scaler.inverse_transform([[pred_norm]])[0,0]
        cur_price = data['Close'].iloc[-1].item()
        stock_rsi = rsi[-1]
        stock_ma20 = ma20[-1]
        decision = advisor_decision(pred_price, cur_price, sentiment, stock_rsi, stock_ma20)
        clear_text_output(ticker, cur_price, pred_price, stock_rsi, stock_ma20, sentiment, decision)
        show_prediction_plot(data, pred_price, cur_price, ticker, decision)
        portfolio.append(data['Close'].pct_change()[-100:].fillna(0))
    if len(portfolio) >= 2:
        port_df = pd.concat([s.reset_index(drop=True) for s in portfolio], axis=1)
        port_df = port_df.fillna(0)
        optimal_w = optimize_portfolio(port_df)
        print("\nOptimized Portfolio Weights:")
        for i, ticker in enumerate(tickers):
            print(f"  {ticker:<10} : {optimal_w[i]:.2%}")
    # Backtest on first ticker
    model, test_results = backtest_lstm_bot(tickers[0], period='6mo', seq_len=seq_len, epochs=6)
    if test_results:
        plot_backtest(test_results, stockname=tickers[0])
    print("\nAnalysis and prediction completed.🟢")

# === MAIN ===
if __name__ == "__main__":
    stock_trading_bot()
