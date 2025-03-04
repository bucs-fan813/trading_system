
1. **Awesome Oscillator**  
   The strategy calculates the median price as  
   \[
   \text{median\_price} = \frac{\text{high} + \text{low}}{2}
   \]  
   Then it computes two simple moving averages (SMAs) on the median price: a short‐term SMA (with period _nₛ_) and a long‐term SMA (with period _nₗ_, where _nₛ_ < _nₗ_). The Awesome Oscillator (AO) is then given by  
   \[
   \text{AO} = \text{SMA}_{nₛ} - \text{SMA}_{nₗ}
   \]  
   Trading signals are generated on the zero–crossing of AO:  
   - A buy signal (signal = 1) is triggered when the AO crosses upward past zero (i.e. previous AO ≤ 0 and current AO > 0).  
   - A sell signal (signal = –1) is triggered when the AO crosses downward past zero (previous AO ≥ 0 and current AO < 0).  
   In “long–only” mode the sell signals are suppressed (converted to 0).  
   The signal “strength” is taken as the normalized absolute AO value  
   \[
   \text{signal\_strength} = \frac{|\text{AO}|}{\text{rolling\_std}(\text{AO}) + \epsilon}
   \]
   where \(\epsilon\) is a small constant (1e–6) to avoid division by zero.
   
2. **Coppock Curve**  
   - **ROC Calculation:** Two rates‐of‐change (ROC) are computed from the closing price using different lookback periods (in days estimated from input months).  
     \[
     ROC1 = 100 \times \left(\frac{\text{Close}}{\text{Close}_{\text{lagged by }roc1\_days}} - 1\right)
     \]
     \[
     ROC2 = 100 \times \left(\frac{\text{Close}}{\text{Close}_{\text{lagged by }roc2\_days}} - 1\right)
     \]
   - **Combined ROC and Smoothing:** The two ROC values are summed, and the combined series is smoothed using a weighted moving average (WMA) with linearly increasing weights. Mathematically, if the weights are \( w_1, w_2, \dots, w_N \) (with \( w_i=i \)), then:
     \[
     \text{WMA} = \frac{w_1 \times x_1 + w_2 \times x_2 + \cdots + w_N \times x_N}{w_1 + w_2 + \cdots + w_N}
     \]
   - **Signal Generation:**  
     - A long signal (+1) is generated when—after a sustained negative regime (e.g. 4 consecutive days with negative Coppock Curve values)—the indicator crosses above zero.  
     - A short signal (-1) is triggered when, after a sustained positive regime, the indicator turns negative.  
     - An additional momentum strength metric is computed as the divergence between the Coppock Curve and a short‐term reference (its 21-day moving average), with an optional normalization to the \([-1, 1]\) range.

3. **Know Sure Thing**  
  The strategy computes the KST indicator as the sum of several weighted moving averages of the rate‐of‐change (ROC) of the close price. In detail, for each ROC component, we compute  
  - **ROC:**  
    \[
    \text{ROC}(t; \text{period}) = \left(\frac{\text{close}_t - \text{close}_{t-\text{period}}}{\text{close}_{t-\text{period}}}\right) \times 100
    \]
  - **Smoothed ROC:** The above ROC is then smoothed using a simple moving average (SMA) over a specified period.  
  - **KST Indicator:**  
    \[
    \text{KST}(t) = \sum_{i=1}^{4} w_i \times \text{SMA}(\text{ROC}(\text{close}, \text{roc\_period}_{i}), \text{sma\_period}_{i})
    \]
  - **Signal Line:** This is computed as the SMA of the KST indicator over a given signal period.  
  A bullish crossover (KST crossing above its signal line) triggers a long signal (or entry) and a bearish crossover triggers an exit (or a short signal if short positions are allowed). In long‑only mode, bearish crossovers lead to exits.

4. **MCAD Strategy**  
   - *Fast EMA and Slow EMA:* The fast EMA (span = fast) and the slow EMA (span = slow) are computed on the close price.
   - *MACD Line:* Defined as `MACD = fast_ema - slow_ema`.
   - *Signal Line:* Calculated as an EMA of the MACD line over the smooth period.
   - *Histogram:* The difference between the MACD line and the signal line indicates the strength and direction of the momentum.

    **Signal Generation:**  
   - A crossover upward (MACD crossing above the signal line) generates a bullish signal (+1).  
   - A crossover downward generates a bearish signal (–1) if not in long-only mode (or 0 if long-only).

5. **Relative Vigor Index**  
  The RVI is computed as  
  \[
  \text{Numerator}(t) = (C(t) - O(t)) + 2 \times (C(t-1) - O(t-1)) + 2 \times (C(t-2) - O(t-2)) + (C(t-3) - O(t-3))
  \]  
  \[
  \text{Denom}(t) = (H(t) - L(t)) + 2 \times (H(t-1) - L(t-1)) + 2 \times (H(t-2) - L(t-2)) + (H(t-3) - L(t-3))
  \]  
  and the raw RVI is  
  \[
  \text{RVI}_\text{raw}(t) = \frac{\text{Numerator}(t)}{\text{Denom}(t)}
  \]  
  which is then smoothed by taking a simple moving average (SMA) over the lookback period. Its signal line is computed as  
  \[
  \text{Signal Line}(t) = \frac{\text{RVI}(t) + 2\,\text{RVI}(t-1) + 2\,\text{RVI}(t-2) + \text{RVI}(t-3)}{6}
  \]  
  Trading signals are generated when a crossover occurs (buy when RVI crosses above the signal line and sell when it crosses below). In “long only” mode sell signals are replaced with 0. Signal strength is computed as the z‑score of the absolute difference between RVI and its signal line over a rolling window.

6. **SMA Crossover**  
  - Computes the short‑term SMA (Sₜ) and long‑term SMA (Lₜ) as:  
    • Sₜ = (1/Nₛₕₒᵣₜ) · Σ (close price over the past Nₛₕₒᵣₜ bars)  
    • Lₜ = (1/Nₗₒₙg) · Σ (close price over the past Nₗₒₙg bars)  
  - Generates a **buy signal** (signal = 1) when a bullish crossover occurs (i.e. Sₜ > Lₜ and the previous period Sₜ₋₁ ≤ Lₜ₋₁) and a **sell signal** (signal = –1) when a bearish crossover occurs (i.e. Sₜ < Lₜ and the previous period Sₜ₋₁ ≥ Lₜ₋₁). Signal strength is calculated as (Sₜ – Lₜ) divided by the current close so that it is normalized.  
  - In “long‑only” mode the sell signals (–1) are overridden to 0 so that only long positions are taken.  


7. **True Strength Index**  
   The True Strength Index (TSI) is computed as follows:  
   - **Delta:**  
     \[
     \Delta = \text{close.diff()}
     \]
   - **Double Smoothed Values:**  
     \[
     \text{double\_smoothed} = EMA(EMA(\Delta, \text{span} = \text{long\_period}), \text{span} = \text{short\_period})
     \]
     \[
     \text{abs\_double\_smoothed} = EMA(EMA(|\Delta|, \text{span} = \text{long\_period}), \text{span} = \text{short\_period})
     \]
   - **TSI and Signal Line:**  
     \[
     TSI = 100 \times \frac{\text{double\_smoothed}}{\text{abs\_double\_smoothed} + 1\times10^{-10}}
     \]
     \[
     \text{signal\_line} = EMA(TSI, \text{span} = \text{signal\_period})
     \]
   - **Signals:**  
     A buy signal (+1) is generated when the TSI crosses above the signal line (i.e. the prior bar was below and the current bar is above), and a sell signal (–1) is generated on the opposite crossover. When in long-only mode, sell signals are forced to 0.  

