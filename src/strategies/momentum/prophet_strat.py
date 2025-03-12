# trading_system/src/strategies/momentum/prophet_forecast.py

"""
Prophet Forecast Strategy with Integrated Risk Management Component.

This strategy uses the Prophet model to forecast future stock prices and generate
a directional trading signal based on multi-horizon predictions. For each ticker,
the model is trained on historical close prices. Forecasts are generated at three
horizons:
  
  - h_short (e.g., 7 days)
  - h_mid   (e.g., 14 days)
  - h_long  (e.g., 30 days)
  
For a given ticker with current price Pₜ:
  - A long signal (signal = 1) is generated if the forecasted prices (Fₛₕₒᵣₜ, Fₘᵢd, Fₗₒₙg)
    are all above Pₜ and strictly rising:
        Fₛₕₒᵣₜ > Pₜ, Fₘᵢd > Pₜ, Fₗₒₙg > Pₜ  and  Fₛₕₒᵣₜ < Fₘᵢd < Fₗₒₙg.
    The signal strength is computed as (Fₗₒₙg − Pₜ) / Pₜ.
    
  - A short signal (signal = -1) is generated if the forecasted prices are all below Pₜ and
    strictly falling:
        Fₛₕₒᵣₜ < Pₜ, Fₘᵢd < Pₜ, Fₗₒₙg < Pₜ  and  Fₛₕₒᵣₜ > Fₘᵢd > Fₗₒₙg.
    The signal strength is computed as (Pₜ − Fₗₒₙg) / Pₜ.
    
  - Otherwise, no signal is produced (signal = 0, strength = 0).

When the full (backtest) DataFrame is generated (i.e. latest_only==False), the constant
signal (and forecast details) are applied over the backtest period (which is taken as the
portion of historical data from the last training date onward) and then risk management is
applied (via the RiskManager class) to simulate entry/exit, stop-loss, and take-profit events,
with vectorized calculations for speed. When latest_only==True, only the last day's signal and
forecast values are returned for a quick EOD decision.

Args:
    tickers (str or List[str]): Stock ticker symbol or list of tickers.
    start_date (str, optional): Start date (YYYY-MM-DD) for historical data retrieval.
    end_date (str, optional): End date (YYYY-MM-DD) for historical data retrieval.
    initial_position (int): Starting trading position (default=0, meaning no position).
    latest_only (bool): If True, returns only the latest prediction row per ticker; when False,
                        a full daily DataFrame (with risk–managed returns) is returned.

Strategy-specific parameters passed in `params` (with defaults):
    - 'horizon_short': Forecast horizon in days for the shortest forecast (default: 7).
    - 'horizon_mid'  : Forecast horizon in days for the mid forecast (default: 14).
    - 'horizon_long' : Forecast horizon in days for the longest forecast (default: 30).
    - 'stop_loss_pct': Stop loss percentage (default: 0.05).
    - 'take_profit_pct': Take profit percentage (default: 0.10).
    - 'slippage_pct'   : Slippage percentage (default: 0.001).
    - 'transaction_cost_pct': Transaction cost percentage (default: 0.001).

Outputs:
    When latest_only is True:
        A DataFrame with one row per ticker containing:
          - 'ticker'
          - 'current_price'
          - 'forecast_short'
          - 'forecast_mid'
          - 'forecast_long'
          - 'signal'
          - 'signal_strength'
          - Price references ('high', 'low', 'close')
    
    When latest_only is False (backtesting mode):
        A full daily DataFrame (for each ticker) with the original price columns plus:
          - 'current_price'
          - 'forecast_short'
          - 'forecast_mid'
          - 'forecast_long'
          - 'signal'
          - 'signal_strength'
        and additional columns resulting from RiskManager such as:
          - 'position', 'return', 'cumulative_return', 'exit_type'
        (the risk management returns are computed in a vectorized fashion.)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from prophet import Prophet
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class ProphetForecast(BaseStrategy):
    """
    Prophet Forecast Strategy for Multi-Horizon Price Prediction with Integrated Risk Management.

    This strategy trains a Prophet model on historical close prices for each given ticker,
    forecasts future prices at short, mid, and long horizons and produces a directional signal:
      - Signal 1 (long): Forecasts are all above the current close and strictly increasing.
      - Signal -1 (short): Forecasts are all below the current close and strictly decreasing.
      - Signal 0 (no signal): Otherwise.

    The signal strength is computed as the relative difference between the longest forecast
    and the current price. The forecast values and signal are then merged with historical price
    data and passed into a RiskManager module that integrates stop-loss, take-profit, slippage,
    and transaction cost adjustments. This allows full backtest simulation (with computed trade
    returns, cumulative returns, etc.) or a quick end-of-day signal assessment.

    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Strategy-specific parameters (see module docstring for details).
    """
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        default_params = {
            'horizon_short': 7,
            'horizon_mid': 14,
            'horizon_long': 30,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
        # Initialize RiskManager with provided risk parameters.
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )

    def generate_signals(self,
                         tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate directional signals via Prophet-based forecasts and then simulate trades
        using integrated risk management.

        For each ticker:
          1. Retrieve historical price data (columns include 'close', 'high', 'low', etc.)
             for the period defined by start_date and end_date.
          2. Prepare the data for Prophet (renaming the date and close columns to 'ds' and 'y').
          3. Train the Prophet model on the historical close prices.
          4. Forecast future prices up to 'horizon_long' days beyond the last training date.
          5. Sample the forecast at target dates (current date + horizon_short, +horizon_mid, +horizon_long).
          6. Compute the current price and decide on the signal:
                - Long (1) if F_short, F_mid, F_long > current price and F_short < F_mid < F_long.
                - Short (-1) if F_short, F_mid, F_long < current price and F_short > F_mid > F_long.
                - Otherwise, 0.
          7. Compute signal strength as the relative difference between the current price and F_long.
          8. If latest_only is True, return only the last row (with forecast details and signal).
             Otherwise, assign the constant signal (and forecast values) to each row of backtest data
             (from the last training date onward) and then apply the RiskManager to simulate trading.

        Args:
            tickers (str or List[str]): Stock ticker symbol or list of ticker symbols.
            start_date (str, optional): Start date (YYYY-MM-DD) for historical data retrieval.
            end_date (str, optional): End date (YYYY-MM-DD) for historical data retrieval.
            initial_position (int): Starting trading position; used by RiskManager (default=0).
            latest_only (bool): If True, return only the latest row per ticker for EOD signal.
                                If False, return a full daily time-series for backtest simulation.

        Returns:
            pd.DataFrame: DataFrame with trading signals, forecast values, risk-managed returns,
                          and full price data to allow downstream performance evaluation.
        """
        # Ensure tickers is a list.
        if isinstance(tickers, str):
            tickers = [tickers]

        # Retrieve historical prices for all tickers in a vectorized manner.
        price_data_all = self.get_historical_prices(tickers, from_date=start_date, to_date=end_date)
        results = []

        for ticker in tickers:
            # For multiple tickers, data is a MultiIndex (ticker, date); extract ticker-specific slice.
            if isinstance(price_data_all.index, pd.MultiIndex):
                try:
                    data = price_data_all.xs(ticker, level='ticker').copy()
                except KeyError:
                    self.logger.warning("No data found for ticker %s", ticker)
                    continue
            else:
                data = price_data_all.copy()

            data.sort_index(inplace=True)
            if data.empty or len(data) < 30:
                self.logger.warning("Insufficient data for %s to generate Prophet forecast.", ticker)
                continue

            # Prepare data for Prophet training.
            df_prophet = data.reset_index()
            # Ensure the date column is named 'ds' (if not already present).
            if 'date' not in df_prophet.columns:
                df_prophet.rename(columns={df_prophet.columns[0]: 'ds'}, inplace=True)
            else:
                df_prophet.rename(columns={'date': 'ds'}, inplace=True)
            if 'close' not in df_prophet.columns:
                self.logger.warning("Missing 'close' prices for %s.", ticker)
                continue
            df_prophet.rename(columns={'close': 'y'}, inplace=True)
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            if len(df_prophet) < 30:
                self.logger.warning("Not enough historical data for %s to reliably train Prophet.", ticker)
                continue

            # Train the Prophet model.
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            # Forecast future prices up to the longest horizon.
            horizon_long = self.params.get('horizon_long', 30)
            future = model.make_future_dataframe(periods=horizon_long, freq='D')
            forecast = model.predict(future)

            # Use the final training date as "current" reference.
            current_price = df_prophet.iloc[-1]['y']
            current_date = df_prophet.iloc[-1]['ds']

            # Helper function to extract forecast value closest to a target date.
            def get_forecast_value(target_date: pd.Timestamp) -> float:
                idx = (forecast['ds'] - target_date).abs().idxmin()
                return forecast.loc[idx, 'yhat']

            horizon_short = self.params.get('horizon_short', 7)
            horizon_mid = self.params.get('horizon_mid', 14)

            target_date_short = current_date + pd.Timedelta(days=horizon_short)
            target_date_mid = current_date + pd.Timedelta(days=horizon_mid)
            target_date_long = current_date + pd.Timedelta(days=horizon_long)

            forecast_short = get_forecast_value(target_date_short)
            forecast_mid = get_forecast_value(target_date_mid)
            forecast_long_val = get_forecast_value(target_date_long)

            # Determine the trading signal.
            if (forecast_short > current_price and
                forecast_mid > current_price and
                forecast_long_val > current_price and
                forecast_short < forecast_mid < forecast_long_val):
                signal = 1
            elif (forecast_short < current_price and
                  forecast_mid < current_price and
                  forecast_long_val < current_price and
                  forecast_short > forecast_mid > forecast_long_val):
                signal = -1
            else:
                signal = 0

            # Compute signal strength.
            if signal == 1:
                signal_strength = (forecast_long_val - current_price) / current_price
            elif signal == -1:
                signal_strength = (current_price - forecast_long_val) / current_price
            else:
                signal_strength = 0

            if latest_only:
                # Return only the latest (last) row of the historical data with forecast details.
                last_row = data.iloc[-1].copy()
                last_row["ticker"] = ticker
                last_row["current_price"] = current_price
                last_row["forecast_short"] = forecast_short
                last_row["forecast_mid"] = forecast_mid
                last_row["forecast_long"] = forecast_long_val
                last_row["signal"] = signal
                last_row["signal_strength"] = signal_strength
                results.append(last_row.to_frame().T)
            else:
                # For backtesting, use data from current_date onward.
                backtest_data = data[data.index >= current_date].copy()
                if backtest_data.empty:
                    self.logger.warning("No backtest data available for %s beyond current date.", ticker)
                    continue
                backtest_data["ticker"] = ticker
                backtest_data["current_price"] = current_price
                backtest_data["forecast_short"] = forecast_short
                backtest_data["forecast_mid"] = forecast_mid
                backtest_data["forecast_long"] = forecast_long_val
                backtest_data["signal"] = signal
                backtest_data["signal_strength"] = signal_strength

                # Risk management is applied to the full daily time-series.
                managed_data = self.risk_manager.apply(backtest_data, initial_position=initial_position)
                results.append(managed_data)

        if results:
            final_df = pd.concat(results, axis=0, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()