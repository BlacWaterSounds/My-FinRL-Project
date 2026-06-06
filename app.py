def run_pipeline():
    # ... (Your entire existing code goes here, indented)
    # Instead of plt.show(), return the df_account_value
    return df_account_value
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.agents.stablebaselines3.models import DRLAgent

print("--- Pipeline Started ---")

# --- 1. SETUP DATA ---
print("--- Downloading Data ---")
data_result = YahooDownloader(start_date='2023-01-01', end_date='2024-01-01', ticker_list=['AAPL']).fetch_data()
df = data_result[0] if isinstance(data_result, tuple) else data_result

# Flatten MultiIndex to standard columns
df.columns = df.columns.map(lambda x: x[0].lower() if isinstance(x, tuple) else str(x).lower())
df = df.reset_index()

# Rename and Clean
rename_map = {'adjclose': 'close', 'ticker': 'tic', 'level_0': 'date'}
df.rename(columns=rename_map, inplace=True)

# Drop duplicates to prevent KeyError in FeatureEngineer
df = df.loc[:, ~df.columns.duplicated()]

# Select essential columns for FinRL
required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
df = df[[c for c in required_cols if c in df.columns]]

# Ensure 'tic' exists
if 'tic' not in df.columns:
    df['tic'] = 'AAPL'

print("Cleaned columns:", df.columns.tolist())

# Feature Engineering
tech_indicators = ['macd', 'rsi_30']
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=tech_indicators)
test_df = fe.preprocess_data(df)
print("Data preprocessing successful. Shape:", test_df.shape)

# --- 2. SETUP ENVIRONMENT ---
# Added num_stock_shares and reward_scaling to fix the TypeError
env_kwargs = {
    "df": test_df, 
    "stock_dim": 1, 
    "hmax": 100, 
    "initial_amount": 100000,
    "buy_cost_pct": [0.001], 
    "sell_cost_pct": [0.001], 
    "state_space": 5, 
    "action_space": 1, 
    "tech_indicator_list": tech_indicators,
    "num_stock_shares": [0],      # Added: Initial shares held
    "reward_scaling": 1e-4        # Added: Scaling factor for rewards
}
e_trade_gym = StockTradingEnv(**env_kwargs)

# --- 3. RUN BACKTEST ---
print("--- Loading Model & Running Backtest ---")
# Ensure 'ppo_aapl_model.zip' is in the same folder
trained_model = PPO.load("ppo_aapl_model")
df_account_value, _ = DRLAgent.DRL_prediction(model=trained_model, environment=e_trade_gym)

# --- 4. VISUALIZATION ---
print("--- Generating and Saving Plot ---")
plt.figure(figsize=(10, 6))
plt.plot(df_account_value['account_value'], label='Agent Portfolio', color='blue')
plt.title('Reinforcement Learning Performance (2023-2024)')
plt.xlabel('Trading Days')
plt.ylabel('Account Value ($)')
plt.legend()
plt.grid(True)

# Save the plot to the current folder
plt.savefig("performance_plot.png")
print("Plot saved successfully as 'performance_plot.png'")