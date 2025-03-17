# Cup and Handle Strategy

## Strategy Overview

The Cup and Handle pattern is a technical chart pattern characterized by a rounded bottom followed by a short consolidation period. 

### Mathematical Framework

Let's define the price series as $P(t)$ over time $t$:

1. **Cup Formation**:
   - Left peak at time $t_1$ with price $P_1 = P(t_1)$
   - Right peak at time $t_2$ with price $P_2 = P(t_2)$
   - Cup bottom with minimum price $P_{bottom}$ between $t_1$ and $t_2$
   - Resistance level $R = \max(P_1, P_2)$
   - Cup depth $D = R - P_{bottom}$
   - Relative cup depth $D_r = \frac{D}{R}$ (must be $\leq$ cup_depth_threshold)

2. **Handle Formation**:
   - Follows the cup's right peak
   - Handle depth $D_h$ (should be $\leq$ handle_depth_threshold $\times$ $D$)
   - Duration between min_handle_duration and max_handle_duration days

3. **Breakout Signal**:
   - Generated when price exceeds resistance: $P(t_{current}) > R \times (1 + \text{breakout_threshold})$
   - Signal strength = $\frac{P(t_{current})}{R} - 1$
   - Optional volume confirmation: $V(t_{current}) > V_{avg}$

4. **Risk Management**:
   - Stop loss: Entry price $\times$ (1 - stop_loss_pct)
   - Take profit: Entry price $\times$ (1 + take_profit_pct)
   - Adjusted for slippage and transaction costs
---

# Triangle Breakout Strategy

## Overview

The Triangle Breakout strategy identifies consolidation patterns (triangles) in price action by fitting two converging trend lines through local highs and lows within a rolling window. When price breaks out of these patterns with sufficient momentum, trading signals are generated.

## Mathematical Formulation

1. **Pattern Detection**: Within a rolling window of `max_lookback` bars, we identify local extrema (peaks and troughs) and fit linear regression lines:

   Upper Trendline: $y_{upper}(x) = m_{upper} \cdot x + c_{upper}$
   
   Lower Trendline: $y_{lower}(x) = m_{lower} \cdot x + c_{lower}$

   where $x$ is the relative index within the window (0 to `max_lookback-1`).

2. **Triangle Classification**:
   - **Symmetrical Triangle**: $m_{upper} < -0.01$ and $m_{lower} > 0.01$
   - **Ascending Triangle**: $|m_{upper}| < 0.01$ and $m_{lower} > 0.01$
   - **Descending Triangle**: $m_{upper} < -0.01$ and $|m_{lower}| < 0.01$

3. **Pattern Validation**:
   - Minimum points requirement: At least `min_points` local extrema for each trendline
   - Pattern significance: Pattern height at $x = max\_lookback-1$ must exceed `min_pattern_size` times the current price

4. **Breakout Detection**:
   - Upward Breakout: $close > y_{upper} \cdot (1 + breakout\_threshold)$
   - Downward Breakout: $close < y_{lower} \cdot (1 - breakout\_threshold)$
   - Optional volume confirmation: $volume > 20day\_average\_volume$

5. **Risk Management**:
   - Stop Loss: Exit position when price moves against the trade by `stop_loss_pct`
   - Take Profit: Exit position when price moves in favor of the trade by `take_profit_pct`
   - Account for slippage and transaction costs in return calculations
---

# Volume Breakout Strategy

The Volume Breakout strategy identifies price movements that break through key resistance or support levels with confirmation from significant volume increases. This approach combines technical analysis with volume confirmation to generate high-probability trading signals.

1. ## Strategy Overview

The strategy works by:

1. Establishing dynamic resistance and support levels based on historical price action
2. Identifying price breakouts beyond these levels
3. Confirming breakouts with volume surges
4. Filtering signals based on volatility (optional)
5. Implementing risk management rules

2. ## Key Calculations
- Resistance level: $\text{Resistance}_t = \max(\text{high}_{t-n+1}, ..., \text{high}_t)$
- Support level: $\text{Support}_t = \min(\text{low}_{t-n+1}, ..., \text{low}_t)$
- Average Volume: $\text{AvgVolume}_t = \frac{1}{m}\sum_{i=t-m+1}^{t} \text{volume}_i$
- Price Change: $\text{PriceChange}_t = \frac{\text{close}_t - \text{close}_{t-1}}{\text{close}_{t-1}}$

3. ## Signal Generation

### Buy Signal (1) Conditions:
1. $\text{close}_t > \text{Resistance}_{t-1}$ for ≥ consecutive_bars bars
2. $\text{volume}_t > \text{volume\_threshold} \times \text{AvgVolume}_t$
3. $|\text{PriceChange}_t| > \text{price\_threshold}$
4. Optional: $|\text{PriceChange}_t| > \text{atr\_threshold} \times \frac{\text{ATR}_t}{\text{close}_t}$

### Sell Signal (-1) Conditions:
1. $\text{close}_t < \text{Support}_{t-1}$ for ≥ consecutive_bars bars
2. $\text{volume}_t > \text{volume\_threshold} \times \text{AvgVolume}_t$
3. $|\text{PriceChange}_t| > \text{price\_threshold}$
4. Optional: $|\text{PriceChange}_t| > \text{atr\_threshold} \times \frac{\text{ATR}_t}{\text{close}_t}$

4. ## Signal Strength
$\text{SignalStrength} = \frac{|\text{PriceChange}_t|}{\text{PriceVolatility} + 10^{-6}}$

Where PriceVolatility is $\frac{\text{ATR}_t}{\text{close}_t}$ if ATR filter is enabled, otherwise $|\text{PriceChange}_t|$.

5. ## Risk Management
- Stop Loss (long): $\text{StopLoss} = \text{EntryPrice} \times (1 - \text{stop\_loss\_pct})$
- Take Profit (long): $\text{TakeProfit} = \text{EntryPrice} \times (1 + \text{take\_profit\_pct})$
- For short positions, these are inverted