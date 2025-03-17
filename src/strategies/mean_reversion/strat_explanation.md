# Commodity Channel Index (CCI) Strategy

The Commodity Channel Index (CCI) is a versatile momentum oscillator that measures the current price level relative to an average price level over a given period. This strategy identifies potential entry and exit points by monitoring when the CCI crosses specific thresholds.

## Mathematical Formulation

The CCI calculation involves several steps:

1. **Typical Price (TP)**: 
   $TP = \frac{High + Low + Close}{3}$

2. **Simple Moving Average (SMA)** of TP over period $n$:
   $SMA = \frac{1}{n} \sum_{i=1}^{n} TP_i$

3. **Mean Absolute Deviation (MAD)** of TP over period $n$:
   $MAD = \frac{1}{n} \sum_{i=1}^{n} |TP_i - SMA|$

4. **Commodity Channel Index (CCI)**:
   $CCI = \frac{TP - SMA}{0.015 \times MAD}$

## Trading Logic

- **Buy Signal**: Generated when the previous CCI is below the lower threshold (`cci_lower_band`) and the current CCI crosses above it.
- **Sell Signal**: Generated when the previous CCI is above the upper threshold (`cci_upper_band`) and the current CCI crosses below it.
- **Signal Strength**: Measured as the difference between the current CCI value and the threshold that was breached.
---

# Disparity Index Strategy

## Strategy Overview

The Disparity Index strategy is a technical indicator-based approach that identifies potential market reversals by measuring the relative deviation of price from its moving average. It's particularly effective for detecting overbought and oversold conditions and can be applied to various timeframes.

## Mathematical Foundation

The Disparity Index (DI) is calculated as:

$$DI = \left(\frac{Close - MA}{MA}\right) \times 100$$

Where:
- $Close$ is the closing price
- $MA$ is the moving average over a specified lookback period

## Signal Generation

Trading signals are generated based on DI transitions:

1. **Buy Signal (Long)**: Triggered when DI becomes positive after being negative for a consecutive number of periods
   - Signal Strength = DI value

2. **Sell Signal (Exit or Short)**: Triggered when DI becomes negative after being positive for a consecutive number of periods
   - Signal Strength = -DI value

3. **Long-only Mode**: When enabled, sell signals are converted to exit signals (0)

## Risk Management

The strategy integrates comprehensive risk management:

- **Entry Adjustment**: Accounts for slippage and transaction costs
- **Exit Conditions**: 
  - Stop-loss triggered at specified percentage below entry price
  - Take-profit activated at specified percentage above entry price
  - Optional trailing stop to lock in profits
  - Signal reversal can also trigger exits

## Performance Calculation

Returns are calculated as:
- For long trades: $\left(\frac{Exit Price}{Entry Price}\right) - 1$
- For short trades: $\left(\frac{Entry Price}{Exit Price}\right) - 1$

Cumulative returns are aggregated across all trades.

## Key Parameters

- `di_lookback`: Period for moving average calculation
- `consecutive_period`: Required number of consecutive days in one direction before signal reversal
- Risk parameters: Stop-loss, take-profit, trailing stop, slippage, and transaction costs

---

# RSI Strategy

## Strategy Overview

The Relative Strength Index (RSI) strategy is a momentum-based technical analysis approach that identifies potential entry and exit points based on overbought and oversold conditions.

## Mathematical Foundation

The RSI is calculated using Wilder's smoothing method:

$$RSI = 100 - \frac{100}{1 + RS}$$

where RS (Relative Strength) is:

$$RS = \frac{EMA(gain, period, \alpha = 1/period)}{EMA(loss, period, \alpha = 1/period)}$$

Gains are positive price differences between consecutive closing prices (with zeros for negative differences), and losses are absolute values of negative differences.

## Signal Generation

- **Buy Signal**: Generated when RSI crosses above the oversold threshold
- **Sell Signal**: Generated when RSI crosses below the overbought threshold (neutralized to 0 in long-only mode)
- **Signal Strength**: Normalized between 0 and 1 based on the distance from the threshold

---

## Stochastic Oscillator Strategy Explanation

The Stochastic Oscillator is a momentum indicator that compares a particular closing price of a security to a range of its prices over a certain period of time. The strategy implemented uses this oscillator to identify potential buy and sell signals based on overbought and oversold conditions.

### Mathematical Foundation

The Stochastic Oscillator consists of two lines: %K (the fast stochastic) and %D (the slow stochastic):

**%K (Fast Stochastic):**
$$\%K = 100 \times \frac{C - L_n}{H_n - L_n}$$

Where:
- $C$ = Current closing price
- $L_n$ = Lowest low for the last $n$ periods
- $H_n$ = Highest high for the last $n$ periods
- $n$ = The lookback period (typically 14 days)

**%D (Slow Stochastic):**
$$\%D = \text{SMA}(\%K, m)$$

Where:
- SMA = Simple Moving Average
- $m$ = Smoothing period (typically 3 days)

### Signal Generation

The strategy generates signals based on the following conditions:

1. **Buy Signal (Long)**: Generated when %K crosses above %D and both are below the oversold threshold.
   * Signal Strength = $\min(1, \frac{\text{oversold} - \min(\%K, \%D)}{\text{oversold}})$

2. **Sell Signal (Short/Exit)**: Generated when %K crosses below %D and both are above the overbought threshold.
   * Signal Strength = $\min(1, \frac{\max(\%K, \%D) - \text{overbought}}{100 - \text{overbought}})$
   * In long-only mode, sell signals are converted to exit signals (0)

---

# Williams %R Strategy

## Overview

The Williams %R (Williams Percent Range) is a momentum oscillator that measures overbought and oversold conditions in the market. It compares a stock's closing price to the high-low range over a specific period, typically 14 days.

## Formula

The Williams %R indicator is calculated as:

$$WR = -100 \times \frac{H_n - C}{H_n - L_n}$$

Where:
- $WR$ is the Williams %R value
- $H_n$ is the highest high over the lookback period $n$
- $L_n$ is the lowest low over the lookback period $n$
- $C$ is the current closing price

## Interpretation

Williams %R oscillates between 0 and -100:
- Values between -80 and -100 indicate oversold conditions (potential buying opportunity)
- Values between 0 and -20 indicate overbought conditions (potential selling opportunity)

## Signal Generation

The strategy generates signals based on the following criteria:

1. **Buy Signal (Long Entry)**: When %R crosses below the oversold threshold (typically -80)
   - Previous %R value is above the oversold threshold
   - Current %R value is below or equal to the oversold threshold

2. **Sell Signal (Long Exit or Short Entry)**: When %R crosses above the overbought threshold (typically -20)
   - Previous %R value is below the overbought threshold
   - Current %R value is above or equal to the overbought threshold

3. **Signal Strength**: Calculated as the absolute difference between the %R value and the relevant threshold
   - For buy signals: $|WR - \text{oversold threshold}|$
   - For sell signals: $|WR - \text{overbought threshold}|$
