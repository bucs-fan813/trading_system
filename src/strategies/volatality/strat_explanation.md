# ATR Trailing Stops Strategy

## Overview
The ATR Trailing Stops strategy is a volatility-based trend following system that uses Average True Range (ATR) to set dynamic trailing stop levels. This approach adapts to changing market conditions by adjusting stop distances based on recent price volatility.

## Key Components

### 1. ATR Calculation
The Average True Range measures market volatility by calculating the greatest of:
- Current high minus current low
- Absolute value of current high minus previous close
- Absolute value of current low minus previous close

$$TR = \max(High - Low, |High - PrevClose|, |Low - PrevClose|)$$

The ATR is then calculated using an exponential moving average with smoothing factor $\alpha = \frac{1}{ATR_{period}}$:

$$ATR = EMA(TR, \alpha)$$

### 2. Trend Determination
A Simple Moving Average (SMA) acts as a trend filter:
- Uptrend: $Close > SMA_{trend\_period}$ (favorable for long positions)
- Downtrend: $Close < SMA_{trend\_period}$ (favorable for short positions)

### 3. Trailing Stop Mechanism

For long positions:
- Entry triggered when trend turns up
- Initial stop distance: $ATR \times ATR_{multiplier}$
- Trailing stop: $TrailingStop_{long} = Close - (ATR \times ATR_{multiplier})$
- Updated stop (ratcheting up): $TrailingStop_{long,t} = \max(TrailingStop_{long,t-1}, Close_t - (ATR_t \times ATR_{multiplier}))$
- Exit when: $Low_t \leq TrailingStop_{long,t-1}$

For short positions:
- Entry triggered when trend turns down
- Initial stop distance: $ATR \times ATR_{multiplier}$
- Trailing stop: $TrailingStop_{short} = Close + (ATR \times ATR_{multiplier})$
- Updated stop (ratcheting down): $TrailingStop_{short,t} = \min(TrailingStop_{short,t-1}, Close_t + (ATR_t \times ATR_{multiplier}))$
- Exit when: $High_t \geq TrailingStop_{short,t-1}$

### 4. Signal Generation
- Buy signal (1): New long position initiated
- Sell signal (-1): New short position initiated
- Exit signal (0): Trailing stop breached
- Signal strength: Normalized distance between price and trailing stop

---

# Bollinger Bands Strategy

## Strategy Overview

The Bollinger Bands strategy implements a mean-reversion trading approach using statistical bands around a moving average to identify potential overbought and oversold conditions.

## Strategy Logic

The strategy calculates three key components:
1. **Simple Moving Average (SMA)**: A rolling average of closing prices over a specified window period.
2. **Standard Deviation (STD)**: Measures price volatility over the same window period.
3. **Bollinger Bands**: Upper and lower bands placed at ±*n* standard deviations from the SMA.

Mathematically, these components are defined as:

$\text{SMA}_t = \frac{1}{w} \sum_{i=t-w+1}^{t} \text{Close}_i$

$\text{STD}_t = \sqrt{\frac{1}{w} \sum_{i=t-w+1}^{t} (\text{Close}_i - \text{SMA}_t)^2}$

$\text{Upper Band}_t = \text{SMA}_t + (m \times \text{STD}_t)$

$\text{Lower Band}_t = \text{SMA}_t - (m \times \text{STD}_t)$

$\text{\%B}_t = \frac{\text{Close}_t - \text{Lower Band}_t}{\text{Upper Band}_t - \text{Lower Band}_t}$

Where:
- $w$ = window period
- $m$ = standard deviation multiplier
- $\text{\%B}_t$ = normalized position of price relative to the bands

## Trade Signals

- **Long Entry**: When price crosses below the lower band (previous close above lower band, current close below lower band)
- **Short Entry**: When price crosses above the upper band (previous close below upper band, current close above upper band)
- The strategy filters out duplicate consecutive signals to ensure only distinct trade entries

---

## GARCH Model Strategy

The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model Strategy is a volatility-based trading approach that generates signals based on forecasted volatility changes. The strategy employs a GARCH(1,1) model to predict future volatility and compares it with historical volatility to determine trading decisions.

### Mathematical Framework

1. **Returns Calculation**:
   - Log returns: $r_t = 100 \times \ln(P_t / P_{t-1})$
   - Simple returns: $r_t = 100 \times (P_t / P_{t-1} - 1)$

2. **GARCH Model**:
   - The GARCH(1,1) model forecasts conditional variance: $\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$
   - A rolling window of size $N$ is used to fit the model

3. **Volatility Comparison**:
   - Forecasted volatility: $\text{forecast\_vol} = \sqrt{\text{forecast\_variance}}$
   - Historical volatility: $\text{hist\_vol} = \text{std}(\text{window\_returns}) \times \sqrt{252}$
   - Relative volatility change: $\text{vol\_change} = \frac{\text{forecast\_vol} - \text{hist\_vol}}{\text{hist\_vol}}$

4. **Signal Generation**:
   - If $\text{vol\_change} > \text{vol\_threshold} \Rightarrow \text{signal} = 1$ (buy)
   - If $\text{vol\_change} < -\text{vol\_threshold} \Rightarrow \text{signal} = -1$ (sell)
   - Otherwise $\text{signal} = 0$ (hold)

5. **Signal Strength**:
   - $\text{signal\_strength} = \frac{|\text{vol\_change}|}{\text{rolling\_std}(\text{vol\_change}) + \epsilon}$

6. **Risk Management**:
   - Positions are adjusted based on stop-loss, take-profit, and trailing stop rules
   - Transaction costs and slippage are accounted for

### Trading Logic

The strategy is based on the principle that significant changes in forecasted volatility provide profitable trading opportunities:
- Rising volatility (positive vol_change) may indicate upcoming market movement, triggering long positions
- Falling volatility (negative vol_change) may signal potential reversals, triggering short positions
- The long_only parameter restricts trading to long positions if set to True

---

# Keltner Channel Strategy Explanation

The Keltner Channel strategy is a volatility-based trading system that uses price action relative to volatility bands to generate trading signals. Here's how it works:

## Core Components

1. **Channel Middle (KC Middle)**: An Exponential Moving Average (EMA) of closing prices
   $$KC_{middle} = EMA(Close, kc\_span)$$

2. **Average True Range (ATR)**: A volatility metric calculated as an EMA of the True Range
   $$TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)$$
   $$ATR = EMA(TR, atr\_span)$$

3. **Channel Boundaries**:
   $$KC_{upper} = KC_{middle} + multiplier \times ATR$$
   $$KC_{lower} = KC_{middle} - multiplier \times ATR$$

## Trading Signals

- **Buy Signal (1)**: Triggered when:
  - Price is below the lower Keltner Channel ($Close < KC_{lower}$)
  - Shows upward momentum ($Close > PrevClose$)
  
- **Sell Signal (-1)**: Triggered when:
  - Price is above the upper Keltner Channel ($Close > KC_{upper}$)
  - Shows downward momentum ($Close < PrevClose$)
  - In long-only mode, sell signals become exit signals (0)

## Signal Strength

- **Buy Signal Strength**: $(KC_{lower} - Close) / ATR$
- **Sell Signal Strength**: $(Close - KC_{upper}) / ATR$

---

# Supertrend Strategy Explanation

## Overview
The Supertrend indicator is a trend-following overlay that uses volatility (ATR) to identify potential trend changes and generate trading signals. It plots a line above or below the price based on the current trend direction, switching sides when the trend reverses.

## Mathematical Formulation

### Step 1: Calculate True Range (TR)
The True Range is the greatest of:
- High - Low
- |High - Previous Close|
- |Low - Previous Close|

$$TR = \max(H-L, |H-C_{prev}|, |L-C_{prev}|)$$

### Step 2: Calculate Average True Range (ATR)
ATR is an exponential moving average of TR over a specified period:

$$ATR = EMA(TR, lookback)$$

### Step 3: Calculate Basic Bands
The basic upper and lower bands are calculated using the ATR and a multiplier:

$$BasicUpper = \frac{H+L}{2} + multiplier \times ATR$$
$$BasicLower = \frac{H+L}{2} - multiplier \times ATR$$

### Step 4: Calculate Final Bands
The final bands are calculated recursively:

**Final Upper Band:**
- For the first period: $FinalUpper_1 = BasicUpper_1$
- For subsequent periods:
$$FinalUpper_t = 
\begin{cases}
BasicUpper_t & \text{if } BasicUpper_t < FinalUpper_{t-1} \text{ or } Close_{t-1} > FinalUpper_{t-1} \\
FinalUpper_{t-1} & \text{otherwise}
\end{cases}$$

**Final Lower Band:**
- For the first period: $FinalLower_1 = BasicLower_1$
- For subsequent periods:
$$FinalLower_t = 
\begin{cases}
BasicLower_t & \text{if } BasicLower_t > FinalLower_{t-1} \text{ or } Close_{t-1} < FinalLower_{t-1} \\
FinalLower_{t-1} & \text{otherwise}
\end{cases}$$

### Step 5: Calculate Supertrend and Trend Direction
- For the first period:
  - If $Close_1 \leq FinalUpper_1$: $Supertrend_1 = FinalUpper_1$ and $Trend_1 = -1$ (bearish)
  - Otherwise: $Supertrend_1 = FinalLower_1$ and $Trend_1 = 1$ (bullish)

- For subsequent periods:
  - If previous trend is bearish ($Trend_{t-1} = -1$):
    - If $Close_t \leq FinalUpper_t$: $Supertrend_t = FinalUpper_t$ and $Trend_t = -1$
    - Otherwise: $Supertrend_t = FinalLower_t$ and $Trend_t = 1$
  - If previous trend is bullish ($Trend_{t-1} = 1$):
    - If $Close_t \geq FinalLower_t$: $Supertrend_t = FinalLower_t$ and $Trend_t = 1$
    - Otherwise: $Supertrend_t = FinalUpper_t$ and $Trend_t = -1$

### Step 6: Generate Trading Signals
- Buy signal (+1): Price crosses above the Supertrend
- Sell signal (-1): Price crosses below the Supertrend
- Signal strength: $SignalStrength = \frac{|Close - Supertrend|}{Close} \times Signal$

---

# Volatility Squeeze Strategy

## Overview

The Volatility Squeeze strategy identifies periods of low volatility followed by potential breakouts. It uses Bollinger Bands (BB) and Keltner Channels (KC) to detect "squeeze" conditions and momentum indicators to determine trade direction.

## Strategy Logic

1. **Squeeze Detection**:
   - A squeeze occurs when Bollinger Bands are contained within Keltner Channels:
     $BB_{lower} > KC_{lower}$ AND $BB_{upper} < KC_{upper}$
   - This indicates unusually low volatility, often preceding significant price moves

2. **Breakout Signal**:
   - A breakout is identified when a squeeze condition ends (squeeze_off = true)
   - The direction is determined by momentum:
     - Long signals: when squeeze_off = true AND momentum > 0
     - Short signals: when squeeze_off = true AND momentum < 0

3. **Mathematical Components**:
   - **Bollinger Bands**:
     $BB_{middle} = MA(close, bb\_period)$
     $BB_{upper} = BB_{middle} + bb\_std \times \sigma(close, bb\_period)$
     $BB_{lower} = BB_{middle} - bb\_std \times \sigma(close, bb\_period)$

   - **Keltner Channels**:
     $KC_{middle} = MA(close, kc\_period)$
     $KC_{upper} = KC_{middle} + kc\_multiplier \times ATR(kc\_atr\_period)$
     $KC_{lower} = KC_{middle} - kc\_multiplier \times ATR(kc\_atr\_period)$

   - **Momentum**:
     Linear regression slope calculated over momentum_period, where:
     $slope = \frac{n \sum(xy) - \sum(x)\sum(y)}{n\sum(x^2) - [\sum(x)]^2}$

---

# Volatility Breakout Strategy

The Volatility Breakout strategy identifies trading opportunities when price breaks out of a volatility-defined range. This approach capitalizes on significant price movements that exceed normal volatility thresholds.

## Strategy Overview

For a given price series $P_t$, the strategy computes the following over a lookback period ($L$):

- **Center Line** ($\mu_t$): Moving average of close prices over $L$ days.
  $$\mu_t = \frac{1}{L}\sum_{i=0}^{L-1}P_{t-i}$$

- **Volatility** ($\sigma_t$): Either rolling standard deviation or Average True Range (ATR).
  - Standard Deviation: $$\sigma_t = \sqrt{\frac{1}{L}\sum_{i=0}^{L-1}(P_{t-i} - \mu_t)^2}$$
  - ATR: Exponential moving average of True Range where:
    $$TR_t = \max[(H_t - L_t), |H_t - C_{t-1}|, |L_t - C_{t-1}|]$$
    $$ATR_t = \alpha \cdot TR_t + (1-\alpha) \cdot ATR_{t-1}$$
    where $\alpha = \frac{1}{period}$

- **Volatility Bands**:
  - Upper Band: $U_t = \mu_t + (m \cdot \sigma_t)$
  - Lower Band: $L_t = \mu_t - (m \cdot \sigma_t)$
  
  where $m$ is the volatility multiplier.

## Signal Generation

- **Buy Signal** (1): When price closes above the upper band: $P_t > U_t$
- **Sell Signal** (-1): When price closes below the lower band: $P_t < L_t$
- **No Signal** (0): When price is between the bands

## Signal Strength

Signal strength measures the magnitude of the breakout:
- For buy signals: $\frac{P_t - U_t}{\sigma_t + \epsilon}$
- For sell signals: $\frac{P_t - L_t}{\sigma_t + \epsilon}$

where $\epsilon$ is a small constant to prevent division by zero.

---