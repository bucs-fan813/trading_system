# Enhanced Market Pressure Strategy

## Overview

The Enhanced Market Pressure Analysis Strategy transforms standard price data into statistical insights by analyzing the normalized position of closing prices within their daily ranges. This approach reveals underlying market forces that conventional indicators might miss.

## Mathematical Foundation

The core of this strategy is modeling the normalized position of closing prices within their daily trading ranges:

$$\text{Normalized Position} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low}}$$

This normalized position is then modeled using probability distributions (primarily Beta distribution) to quantify buying and selling pressures.

## Key Components

1. **Normalized Position Modeling**:
   - Values range from 0 (close at the low) to 1 (close at the high)
   - Values near 1 indicate buying pressure; values near 0 indicate selling pressure

2. **Statistical Distribution Fitting**:
   - Beta distribution fitting: $f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$
   - Parameters derived using method of moments:
     $$\alpha = \mu \cdot \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right)$$
     $$\beta = (1-\mu) \cdot \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right)$$

3. **Pressure Metrics Calculation**:
   - Buying Pressure: $1 - \text{Beta CDF}(0.5; \alpha, \beta)$
   - Selling Pressure: $\text{Beta CDF}(0.5; \alpha, \beta)$
   - Market Pressure: Buying Pressure - Selling Pressure

4. **Signal Generation**:
   - Buy when Market Pressure crosses above threshold
   - Sell when Market Pressure crosses below negative threshold
   - Enhanced by divergence detection between price trends and pressure trends

5. **Statistical Validation**:
   - Signal confidence based on Kolmogorov-Smirnov test
   - Ensures signals are statistically significant

## Advantages

- Captures subtle shifts in market sentiment that may precede price movements
- Volume-weighted analysis for higher confidence in signals
- Statistical significance testing reduces false signals
- Can adapt to different market conditions through parameter optimization



## GARCH-X Strategy: 

### Overview

The GARCH-X Strategy leverages a blend of time-series analysis, volatility modeling, and machine learning (via PCA) to generate trading signals. It aims to capture market dynamics by forecasting future returns based on historical price and volume data, adjusted by risk measures.

### Mathematical Foundation

The core of this strategy lies in modeling returns using a GARCH framework, augmented with exogenous regressors to capture market dependencies.

### Key Components

1.  **Data Preprocessing**:
    -   Log Returns: \( r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \), where \( P_t \) is the closing price at time \( t \).
    -   Volume Standardization: \( z_t = \frac{\ln(V_t) - \mu_t}{\sigma_t} \), where \( V_t \) is the volume at time \( t \), and \( \mu_t \) and \( \sigma_t \) are rolling mean and standard deviation of log volume, respectively.
    -   Price Range: \( hlr_t = \frac{H_t - L_t}{P_{t-1}} \), where \( H_t \) and \( L_t \) are high and low prices at time \( t \).

2.  **Exogenous Regressor Selection**:
    -   Lagged features \( r_{t-i}, z_{t-i}, hlr_{t-i} \) for \( i \in \{1, 2, 3, 4, 5\} \).
    -   PCA applied to reduce dimensionality of exogenous variables, \( \mathbf{x}_t \).

3.  **EGARCH-X Model**:
    -   Conditional Mean: \( r_t = \mu + \mathbf{\beta}^T \mathbf{x}_t + \epsilon_t \), where \( \mathbf{\beta} \) is the coefficient vector for exogenous regressors, and \( \epsilon_t \) is the error term.
    -   Conditional Variance:
        -   EGARCH(p, q) model:
            $$
            \begin{aligned}
            \sigma_t^2 &= \omega + \sum_{i=1}^{p} \alpha_i g(\epsilon_{t-i}) + \sum_{j=1}^{q} \beta_j \sigma_{t-j}^2 \\
            g(\epsilon_{t-i}) &= \theta \epsilon_{t-i} + \gamma [|\epsilon_{t-i}| - E(|\epsilon_{t-i}|)]
            \end{aligned}
            $$
    -   Error Distribution: \( \epsilon_t \sim t(\nu) \), where \( \nu \) is the degrees of freedom.

4.  **Forecasting**:
    -   \( h \)-day forecast of log-returns: \( \hat{r}_{t+1}, \dots, \hat{r}_{t+h} \).
    -   Cumulative Return: \( R = \exp\left(\sum_{i=1}^{h} \hat{r}_{t+i}\right) - 1 \).

5.  **Signal Generation**:
    -   Risk-adjusted Return: \( ra = \frac{R}{\sigma_{\text{forecast}}} \), where \( \sigma_{\text{forecast}} \) is forecast volatility.
    -   Signal Strength: \( ss = ra \cdot z_{\text{last}} \).
    -   Signal:
        -   Buy (\(+1\)) if \( ra > 0 \) and \( z_{\text{last}} > \text{threshold} \).
        -   Sell (\(-1\)) if \( ra < 0 \) and \( z_{\text{last}} > \text{threshold} \) (if not long-only).
        -   Neutral (\(0\)) otherwise.

### Advantages

-   Combines volatility modeling with exogenous factors for enhanced forecasting.
-   Risk-adjusted signals reduce exposure during volatile periods.
-   PCA reduces the impact of multicollinearity.

---

## Prophet Momentum Strategy:

### Overview

The Prophet Momentum Strategy blends time series forecasting using Facebook's Prophet library with trend confirmation through slope analysis. It dynamically adapts to market conditions and volatility regimes to generate trading signals.

### Mathematical Foundation

This strategy relies on the decomposition of time-series data into trend, seasonality, and holiday effects, forecast using Prophet, and validated through trend-slope calculations.

### Key Components

1.  **Data Preparation**:
    -   Input: Price data \( P_t \) (Close), Volume \( V_t \).
    -   Extra Regressors: Volume, Open, High, Low, 14-day RSI.

2.  **Prophet Model**:
    -   Time-series Decomposition: \( y(t) = g(t) + s(t) + h(t) + \epsilon_t \), where \( g(t) \) is trend, \( s(t) \) is seasonality, \( h(t) \) is holiday effects, and \( \epsilon_t \) is the error term.
    -   Trend \( g(t) \): Piecewise linear or logistic growth.
    -   Seasonality \( s(t) \): Modeled via Fourier series:
        $$
        s(t) = \sum_{n=1}^{N} \left[ a_n \cos\left(\frac{2\pi nt}{P}\right) + b_n \sin\left(\frac{2\pi nt}{P}\right) \right]
        $$

3.  **Hyperparameter Tuning via Cross-Validation**:
    -   Optimization Goal: Minimize \( \text{loss} = 1 - \text{DA} \), where \( \text{DA} \) is directional accuracy.
    -   Directional Accuracy (DA): Fraction of forecasts with correct sign prediction.

4.  **Slope Test**:
    -   Forecast Points: \( x = [7, 14, 30] \) (days).
    -   Forecast Values: \( y = [\hat{P}\_7, \hat{P}\_{14}, \hat{P}\_{30}] \).
    -   Slope Calculation: \( \beta = \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(x)} \)
    -   T-Statistic: \( t = \frac{\beta}{\text{std\_err}} \)

5.  **Market Regime Detection**:
    -   Volatility Calculation: 21-day rolling standard deviation of returns \( \sigma_t \).
    -   Hidden Markov Model (HMM) states based on \( \sigma_t \).

6.  **Signal Generation**:
    -   Conditions:
        -   Short Horizon Forecast > Current Price
        -   Long Horizon Forecast > Mid Horizon Forecast
        -   \( \lvert t \rvert > 2 \) (Significant trend)
    -   Signal Strength: \( \text{Signal Strength} = \frac{\lvert f_{\text{long}} - P_t\rvert}{P_t} \times \frac{\lvert t\rvert}{2} \)

### Advantages

-   Combines time-series decomposition with trend confirmation.
-   Dynamic hyperparameter tuning adapts to market changes.
-   Volatility regime detection helps reduce false positives during high-volatility periods.
