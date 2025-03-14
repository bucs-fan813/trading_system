# Trading Strategies Explanation

This document provides an overview of the trading strategies implemented in this repository. For each strategy, the key formulas and signal generation logic are explained. **Note:** If you use this Markdown file on GitHub, keep in mind that the LaTeX math (delimited by `$$`) may not render by default. To view well‐rendered math, consider using GitHub Pages with MathJax/KaTeX enabled or a compatible Markdown viewer.

---

## 1. Awesome Oscillator

The strategy calculates the median price as:

$$
\text{median\_price} = \frac{\text{high} + \text{low}}{2}
$$

Then, two simple moving averages (SMAs) are computed on the median price:
- A **short–term SMA** (with period $n_{s}$)
- A **long–term SMA** (with period $n_{l}$, where $n_{s} < n_{l}$)

The Awesome Oscillator (AO) is defined as:

$$
\text{AO} = \text{SMA}_{n_{s}} - \text{SMA}_{n_{l}}
$$

**Signal Generation:**
- **Buy signal (1):** Triggered when the AO crosses upward past zero (i.e. previous AO ≤ 0 and current AO > 0).
- **Sell signal (–1):** Triggered when the AO crosses downward past zero (previous AO ≥ 0 and current AO < 0).

In “long–only” mode, the sell signals are suppressed (converted to 0).

The signal **strength** is the normalized absolute AO value:

$$
\text{signal\_strength} = \frac{|\text{AO}|}{\text{rolling\_std}(\text{AO}) + \epsilon}
$$

where $\epsilon$ is a small constant (1e–6) to avoid division by zero.

---

## 2. Coppock Curve

**ROC Calculation:**  
Two rates–of–change (ROC) are computed from the closing price using different lookback periods (in days estimated from input months):

$$
ROC1 = 100 \times \left(\frac{\text{Close}}{\text{Close}_{\text{lagged by }\,\text{roc1\_days}}} - 1\right)
$$

$$
ROC2 = 100 \times \left(\frac{\text{Close}}{\text{Close}_{\text{lagged by }\,\text{roc2\_days}}} - 1\right)
$$

**Combined ROC and Smoothing:**  
The two ROC values are summed, and the combined series is smoothed using a weighted moving average (WMA) with linearly increasing weights. Mathematically, if the weights are \(w_1, w_2, \dots, w_N\) (with \(w_i = i\)), then:

$$
\text{WMA} = \frac{w_1 \times x_1 + w_2 \times x_2 + \cdots + w_N \times x_N}{w_1 + w_2 + \cdots + w_N}
$$

**Signal Generation:**
- A **long signal** (+1) is generated when — after a sustained negative regime (e.g. 4 consecutive days with negative Coppock Curve values) — the indicator crosses above zero.
- A **short signal** (–1) is triggered when, after a sustained positive regime, the indicator turns negative.
- Additionally, a momentum strength metric is computed as the divergence between the Coppock Curve and a short–term reference (its 21-day moving average), with an optional normalization to the \([-1, 1]\) range.

---

## 3. Know Sure Thing (KST)

The KST indicator is computed as the sum of several weighted moving averages of the rate–of–change (ROC) of the close price. In detail, for each ROC component:

- **ROC:**

  $$
  \text{ROC}(t; \text{period}) = \left(\frac{\text{close}_t - \text{close}_{t-\text{period}}}{\text{close}_{t-\text{period}}}\right) \times 100
  $$

- **Smoothed ROC:**  
  The above ROC is smoothed using a simple moving average (SMA) over a specified period.

- **KST Indicator:**

  $$
  \text{KST}(t) = \sum_{i=1}^{4} w_i \times \text{SMA}\Big( \text{ROC}\big(\text{close}, \text{roc\_period}_{i}\big), \text{sma\_period}_{i}\Big)
  $$

- **Signal Line:**  
  Computed as the SMA of the KST indicator over a given signal period.

A bullish crossover (KST crossing above its signal line) triggers a long signal (or entry), while a bearish crossover triggers an exit (or a short signal if short positions are allowed). In long‑only mode, bearish crossovers lead only to exits.

---

## 4. MCAD Strategy

- **Fast EMA and Slow EMA:**  
  The fast EMA (with span = `fast`) and the slow EMA (with span = `slow`) are computed on the close price.

- **MACD Line:**  

  $$
  \text{MACD} = \text{fast\_ema} - \text{slow\_ema}
  $$

- **Signal Line:**  
  Calculated as an EMA of the MACD line over a specified smoothing period.

- **Histogram:**  
  The difference between the MACD line and the signal line indicates the strength and direction of the momentum.

**Signal Generation:**
- A crossover upward (MACD crossing above the signal line) generates a bullish signal (+1).
- A crossover downward generates a bearish signal (–1) if not in long‑only mode (or 0 if long‑only).

---

## 5. Relative Vigor Index (RVI)

The RVI is computed as follows:

- **Numerator:**

  $$
  \text{Numerator}(t) = \big( C(t) - O(t) \big) + 2 \times \big( C(t-1) - O(t-1) \big) + 2 \times \big( C(t-2) - O(t-2) \big) + \big( C(t-3) - O(t-3) \big)
  $$

- **Denominator:**

  $$
  \text{Denom}(t) = \big( H(t) - L(t) \big) + 2 \times \big( H(t-1) - L(t-1) \big) + 2 \times \big( H(t-2) - L(t-2) \big) + \big( H(t-3) - L(t-3) \big)
  $$

- **Raw RVI:**

  $$
  \text{RVI}_{\text{raw}}(t) = \frac{\text{Numerator}(t)}{\text{Denom}(t)}
  $$

The raw RVI is then smoothed by taking a simple moving average (SMA) over the lookback period. Its signal line is computed as:

$$
\text{Signal Line}(t) = \frac{\text{RVI}(t) + 2\,\text{RVI}(t-1) + 2\,\text{RVI}(t-2) + \text{RVI}(t-3)}{6}
$$

**Signal Generation:**
- A **buy signal** is generated when the RVI crosses above its signal line.
- A **sell signal** (–1) is generated when the RVI crosses below its signal line.
- In “long‑only” mode, sell signals are replaced with 0.

The signal strength is computed as the z‑score of the absolute difference between the RVI and its signal line over a rolling window.

---

## 6. SMA Crossover

- **SMA Calculations:**
  - **Short–term SMA ($S_t$):**

    $$
    S_t = \frac{1}{N_{\text{short}}} \sum \text{(close price over the past } N_{\text{short}} \text{ bars)}
    $$

  - **Long–term SMA ($L_t$):**

    $$
    L_t = \frac{1}{N_{\text{long}}} \sum \text{(close price over the past } N_{\text{long}} \text{ bars)}
    $$

- **Signal Generation:**
  - **Buy signal (1):**  
    Generated when a bullish crossover occurs (i.e., $S_t > L_t$ and the previous period $S_{t-1} \leq L_{t-1}$).
  - **Sell signal (–1):**  
    Generated when a bearish crossover occurs (i.e., $S_t < L_t$ and the previous period $S_{t-1} \geq L_{t-1}$).

Signal strength is calculated as the normalized difference:

$$
\text{signal strength} = \frac{S_t - L_t}{\text{close}}
$$

In “long‑only” mode, sell signals (–1) are overridden to 0.

---

## 7. True Strength Index (TSI)

The True Strength Index (TSI) is computed as follows:

- **Delta:**

  $$
  \Delta = \text{close.diff()}
  $$

- **Double Smoothed Values:**
  - **Double smoothed delta:**

    $$
    \text{double\_smoothed} = EMA\Big( EMA\big(\Delta,\, \text{span} = \text{long\_period} \big),\, \text{span} = \text{short\_period} \Big)
    $$

  - **Double smoothed absolute delta:**

    $$
    \text{abs\_double\_smoothed} = EMA\Big( EMA\big(|\Delta|,\, \text{span} = \text{long\_period} \big),\, \text{span} = \text{short\_period} \Big)
    $$

- **TSI and Signal Line:**

  $$
  TSI = 100 \times \frac{\text{double\_smoothed}}{\text{abs\_double\_smoothed} + 1\times10^{-10}}
  $$

  $$
  \text{signal\_line} = EMA\Big( TSI,\, \text{span} = \text{signal\_period} \Big)
  $$

- **Signals:**
  - **Buy signal (+1):** Generated when the TSI crosses above its signal line (i.e., the prior bar was below and the current bar is above).
  - **Sell signal (–1):** Generated when the TSI crosses below its signal line.  
    In long‑only mode, sell signals are forced to 0.

---