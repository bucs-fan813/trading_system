# ADX Strategy

- **Mathematical Logic:**  
  The strategy calculates the Average Directional Index (ADX) along with the plus and minus Directional Indicators (DI+ and DI–) as follows:  

  1. **True Range (TR):**  
     For every period,  
     \[
     TR = \max\Big\{(\text{high} - \text{low}),\, |\text{high} - \text{close}_{prev}|,\, |\text{low} - \text{close}_{prev}|\Big\}
     \]
  
  2. **Directional Movements:**  
     \[
     \text{plusDM} = \begin{cases}
     \text{high}_i - \text{high}_{i-1} & \text{if } (\text{high}_i - \text{high}_{i-1}) > (\text{low}_{i-1} - \text{low}_i) \ \text{and} \ > 0 \\
     0 & \text{otherwise}
     \end{cases}
     \]
     \[
     \text{minusDM} = \begin{cases}
     \text{low}_{i-1} - \text{low}_i & \text{if } (\text{low}_{i-1} - \text{low}_i) > (\text{high}_i - \text{high}_{i-1}) \ \text{and} \ > 0 \\
     0 & \text{otherwise}
     \end{cases}
     \]
  
  3. **Wilder’s Smoothing:**  
     The code uses a simple iterative smoothing (effectively an exponential moving average with \( \alpha = \frac{1}{\text{lookback}} \)) for TR, plusDM, and minusDM.  
     Then the smoothed values yield:  
     \[
     \text{DI+} = 100 \times \frac{\text{smooth}(\text{plusDM})}{ATR+\epsilon} \quad,\quad
     \text{DI–} = 100 \times \frac{\text{smooth}(\text{minusDM})}{ATR+\epsilon}
     \]
     The small \(\epsilon\) prevents division by zero.
  
  4. **Directional Index (DX) and ADX:**  
     \[
     DX = 100 \times \frac{|\text{DI+} - \text{DI–}|}{\text{DI+} + \text{DI–}+\epsilon} \quad,\quad ADX = \text{smooth}(DX)
     \]
  
  5. **Signal Generation:**  
     A new trade signal is generated when ADX “crosses above” 25 (i.e. on a transition from below 25 to at or above 25):
     - **Long Signal:** when additionally, DI+ > DI–.
     - **Short Signal:** when DI– > DI+.
     
     Signal strength is defined as:
     \[
     \text{signal\_strength} = \frac{(\text{DI+}-\text{DI–})\times ADX}{100}
     \]

I'll analyze the Aroon strategy, explain it in markdown, and provide a parameter search space for Indian market stocks.
---
# Aroon Strategy

The Aroon strategy is a technical analysis approach that uses the Aroon indicators (Aroon Up and Aroon Down) to identify trend changes and strength in price movements.

## Mathematical Definition

The strategy computes the Aroon indicators over a specified lookback window (default 25 days) as follows:

- **Aroon Up**: $AroonUp = \frac{(days\_since\_highest\_high + 1)}{N} \times 100$
- **Aroon Down**: $AroonDown = \frac{(days\_since\_lowest\_low + 1)}{N} \times 100$

Where:
- $N$ is the lookback period
- $days\_since\_highest\_high$ is the number of periods since the highest high
- $days\_since\_lowest\_low$ is the number of periods since the lowest low

From these, the **Aroon Oscillator** is derived:
$AroonOscillator = AroonUp - AroonDown$

A normalized **Signal Strength** is calculated as:
$SignalStrength = \frac{|AroonOscillator|}{rolling\_std(AroonOscillator) + \epsilon}$

Where $\epsilon$ is a small value (1e-6) to prevent division by zero.

## Trading Signals

- **Long Signal (1)**: When Aroon Up ≥ 70 and Aroon Down ≤ 30
- **Short Signal (-1)**: When Aroon Up ≤ 30 and Aroon Down ≥ 70 (if long_only is False; otherwise, 0)

Signals are forward-filled until a reversal occurs. Risk management is applied using stop-loss, take-profit thresholds, and accounting for transaction costs and slippage.

## Interpretation

- High Aroon Up values (≥70) indicate strong upward trend
- High Aroon Down values (≥70) indicate strong downward trend
- When Aroon Up exceeds Aroon Down significantly, bullish conditions exist
- When Aroon Down exceeds Aroon Up significantly, bearish conditions exist
- Crossovers of these indicators can signal potential trend reversals


I'll analyze the Parabolic SAR strategy code and explain it in markdown format with equations, then provide a parameter search space suitable for Indian markets.

---

I'll analyze the code for the Choppiness Index Strategy, explain it in a concise markdown format, and create a parameter search space suitable for Indian markets.

# Choppiness Index Strategy

## Overview

The Choppiness Index Strategy uses the Choppiness Index (CI) and MACD indicators to identify trending market conditions and generate trading signals. The strategy aims to enter trades when the market is trending (low choppiness) and MACD signals confirm the direction.

## Key Components

### 1. Indicator Calculation

- **Choppiness Index (CI)**:
  $$CI = 100 \times \frac{\log_{10}(\sum_{i=1}^{n} TR_i)}{\log_{10}(n) \times (\max(High, n) - \min(Low, n))}$$
  
  Where:
  - $TR$ = True Range: $\max(High-Low, |High-PrevClose|, |Low-PrevClose|)$
  - $n$ = Period for calculation (default: 14)

- **MACD**:
  $$MACD = EMA_{fast}(Close) - EMA_{slow}(Close)$$
  $$Signal = EMA_{signal}(MACD)$$
  $$Histogram = MACD - Signal$$

### 2. Signal Generation

- **Bullish Signal**: When MACD > Signal Line AND CI < 38.2
- **Bearish Signal**: When MACD < Signal Line AND CI < 38.2
- **Signal Strength**: $Strength = \frac{|Histogram|}{Close} \times (1 - \frac{CI}{38.2})$

---
I'll analyze the Donchian Channel Breakout Strategy and create a strategy explanation in markdown with equations, followed by an appropriate parameter search space for Indian markets.

# Donchian Channel Breakout Strategy

## Strategy Overview

The Donchian Channel Breakout strategy is a trend-following method that generates trading signals based on price breakouts from historical high and low ranges. The strategy uses two sets of channels:

1. **Entry Channels**: Define when to enter long or short positions
2. **Exit Channels**: Define when to exit existing positions

## Mathematical Formulation

For each time point $t$ and ticker:

### Channel Calculations
- **Entry Channels**:
  - Upper Entry Level: $U_e(t) = \max(\text{high}[t-n_e:t-1])$
  - Lower Entry Level: $L_e(t) = \min(\text{low}[t-n_e:t-1])$
  - Middle Channel: $M(t) = \frac{U_e(t) + L_e(t)}{2}$

- **Exit Channels**:
  - Upper Exit Level: $U_x(t) = \max(\text{high}[t-n_x:t-1])$
  - Lower Exit Level: $L_x(t) = \min(\text{low}[t-n_x:t-1])$

### Signal Generation
- **Long Entry**: $\text{signal}(t) = 1$ if $\text{close}(t) > U_e(t)$ and previous position ≤ 0
- **Short Entry**: $\text{signal}(t) = -1$ if $\text{close}(t) < L_e(t)$ and previous position ≥ 0
- **Exit Long**: If position = 1 and $\text{close}(t) < L_x(t)$
- **Exit Short**: If position = -1 and $\text{close}(t) > U_x(t)$

### Optional ATR Filter
When enabled, entries are only valid if:
- Channel Width: $W(t) = U_e(t) - L_e(t)$
- ATR Condition: $W(t) > \text{ATR}(t) \times \text{threshold}$
---
I'll analyze the Ichimoku Cloud strategy code and create a markdown explanation with equations, followed by a suitable parameter search space for Indian markets.

## Ichimoku Cloud Strategy Explanation

The Ichimoku Cloud (Ichimoku Kinko Hyo) is a comprehensive technical analysis system that provides information about support, resistance, trend direction, momentum, and potential trading signals. It consists of five components:

### Key Components

1. **Tenkan-sen (Conversion Line)**: 
   $$\text{Tenkan-sen} = \frac{\text{Highest High} + \text{Lowest Low}}{2} \text{ over } \text{tenkan\_period (default: 9 days)}$$

2. **Kijun-sen (Base Line)**:
   $$\text{Kijun-sen} = \frac{\text{Highest High} + \text{Lowest Low}}{2} \text{ over } \text{kijun\_period (default: 26 days)}$$

3. **Senkou Span A (Leading Span A)**:
   $$\text{Senkou Span A} = \frac{\text{Tenkan-sen} + \text{Kijun-sen}}{2} \text{, shifted forward by displacement (default: 26 days)}$$

4. **Senkou Span B (Leading Span B)**:
   $$\text{Senkou Span B} = \frac{\text{Highest High} + \text{Lowest Low}}{2} \text{ over } \text{senkou\_b\_period (default: 52 days)} \text{, shifted forward by displacement}$$

5. **Chikou Span (Lagging Span)**:
   $$\text{Chikou Span} = \text{Close price shifted backward by displacement}$$

The area between Senkou Span A and Senkou Span B forms the "cloud" (Kumo).

### Signal Generation

The strategy generates signals based on three types of crossovers:

1. **TK Cross**: When Tenkan-sen crosses above Kijun-sen (bullish) or below Kijun-sen (bearish)
2. **Price-Kijun Cross**: When price crosses above Kijun-sen (bullish) or below Kijun-sen (bearish)
3. **Cloud Breakout**: When price crosses above the cloud's upper boundary (bullish) or below the cloud's lower boundary (bearish)

### Entry and Exit Rules

- **Long Entry** (Signal = 1): Triggered when any enabled condition occurs:
  - Tenkan-sen crosses above Kijun-sen
  - Price crosses above Kijun-sen
  - Price crosses above the cloud's upper boundary

- **Short Entry** (Signal = -1): Triggered when any enabled condition occurs:
  - Tenkan-sen crosses below Kijun-sen
  - Price crosses below Kijun-sen
  - Price crosses below the cloud's lower boundary

- **Exit Rules**: Positions are closed based on:
  - Signal reversal
  - Stop-loss being hit (default: 5%)
  - Take-profit being hit (default: 10%)
  - Trailing stop being hit (if enabled)
---
# Parabolic SAR (PSAR) Strategy

The Parabolic SAR (Stop and Reverse) is a trend-following indicator designed to identify potential reversals in price momentum. It appears as dots placed above or below the price, indicating the current trend direction and potential reversal points.

## Mathematical Foundation

The PSAR indicator is computed recursively as follows:

For an **uptrend** (dots below the price):
$$\text{PSAR}_{t} = \text{PSAR}_{t-1} + \text{AF}_{t-1} \times (\text{EP}_{t-1} - \text{PSAR}_{t-1})$$

Where PSAR is bounded by the minimum of the previous two lows.

For a **downtrend** (dots above the price):
$$\text{PSAR}_{t} = \text{PSAR}_{t-1} - \text{AF}_{t-1} \times (\text{PSAR}_{t-1} - \text{EP}_{t-1})$$

Where PSAR is bounded by the maximum of the previous two highs.

Key components:
- **EP (Extreme Point)**: The highest high for an uptrend or the lowest low for a downtrend
- **AF (Acceleration Factor)**: Begins at an initial value and increases by a step amount each time a new EP is recorded, up to a maximum value

## Signal Generation

1. **Trend Identification**: The PSAR determines the current trend (1 for uptrend, -1 for downtrend)
2. **Reversal Detection**: A trend reversal occurs when:
   - In an uptrend: Current low falls below the PSAR
   - In a downtrend: Current high rises above the PSAR
3. **Signal Rules**:
   - **Buy Signal (1)**: Generated when trend changes from downtrend to uptrend
   - **Sell Signal (-1)**: Generated when trend changes from uptrend to downtrend (if not long-only)
   - **Optional ATR Filter**: Validates that price movement is significant relative to volatility

## Signal Strength Calculation

Signal strength provides a measure of conviction, calculated as:
$$\text{SignalStrength} = \text{Signal} \times \frac{|\text{Close} - \text{PSAR}|}{\text{ATR} + \epsilon}$$

Where ε is a small value to avoid division by zero.
