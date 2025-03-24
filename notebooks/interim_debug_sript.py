import sys
import os

# Determine the absolute path to the root directory (assuming the script is one level down)
root_path = os.path.abspath(os.getcwd())

# Add the root directory to sys.path if it’s not already there
if root_path not in sys.path:
    sys.path.insert(0, root_path)

#########################################################
from src.database.config import DatabaseConfig
from src.database.engine import create_db_engine
db_config = DatabaseConfig.default()

#########################################################
import pandas as pd

tickers = pd.read_excel("data/ticker.xlsx")
tickers = tickers[~tickers["Security Name"].duplicated()]
tickers.reset_index(drop=True, inplace=True)

def add_ticker_suffix(x):
    if x["Exchange"]=="BSE":
        return x["Security Name"] + ".BO"
    else:
        return x["Security Name"] + ".NS"

tickers["Ticker"] = tickers.apply(add_ticker_suffix, axis = 1)
all_tickers = tickers["Ticker"].tolist()


#########################################################

from src.strategies.momentum.mcad_strat import MACDStrategy

strategy = MACDStrategy(db_config)

a = strategy.generate_signals(all_tickers[0])