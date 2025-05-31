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



---

# GARCH-X Trading Strategy: Mathematical Explanation

This document provides a comprehensive mathematical explanation of the GARCH-X trading strategy implementation, detailing its key components, statistical foundations, and implementation logic.

## 1. Strategy Overview

The GARCH-X strategy implements an advanced volatility modeling approach that:
- Uses rolling forecast origins with EGARCH-X models
- Incorporates exogenous variables for improved prediction
- Applies PCA for dimension reduction
- Features risk-adjusted signal generation
- Includes dynamic model selection via AIC criterion
- Provides volatility-based position sizing

## 2. Mathematical Foundations

### 2.1 Feature Engineering

#### Log Returns
Daily log returns are calculated as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

where $P_t$ is the closing price at time $t$.

#### Volume Standardization
Volume is log-transformed and standardized using a 20-day rolling window:

$$\text{vol\_log}_t = \ln(\text{volume}_t)$$
$$\text{vol\_mean}_t = \frac{1}{20}\sum_{i=t-19}^{t}\text{vol\_log}_i$$
$$\text{vol\_std}_t = \sqrt{\frac{1}{20}\sum_{i=t-19}^{t}(\text{vol\_log}_i - \text{vol\_mean}_t)^2}$$
$$\text{vol\_z}_t = \frac{\text{vol\_log}_t - \text{vol\_mean}_t}{\text{vol\_std}_t}$$

#### Price Range Features
Two range-based features are computed:

$$\text{hl\_range}_t = \frac{\text{high}_t - \text{low}_t}{\text{close}_{t-1}}$$
$$\text{oc\_range}_t = \frac{|\text{open}_t - \text{close}_t|}{\text{close}_{t-1}}$$

#### Volatility Proxies

**Parkinson Volatility**  
Using high-low range data:

$$\sigma_{P,t} = \sqrt{\frac{1}{4\ln(2)}\cdot \frac{1}{20}\sum_{i=t-19}^{t} \ln^2\left(\frac{\text{high}_i}{\text{low}_i}\right)}$$

**Yang-Zhang Volatility (Simplified)**  
Combining overnight and intraday volatility components:

$$\text{overnight\_ret}_t = \ln\left(\frac{\text{open}_t}{\text{close}_{t-1}}\right)$$
$$\text{oc\_ret}_t = \ln\left(\frac{\text{close}_t}{\text{open}_t}\right)$$

$$\sigma_{\text{YZ},t} = \sqrt{\text{Var}_{20}(\text{overnight\_ret}_t) + \text{Var}_{20}(\text{oc\_ret}_t) + 0.5 \cdot \text{Var}_{20}(\text{hl\_range}_t)}$$

where $\text{Var}_{20}$ denotes the variance over a 20-day rolling window.

### 2.2 EGARCH-X Model

The model combines an ARX specification for the mean equation with an EGARCH specification for the volatility:

**Mean Equation:**
$$r_t = \mu + \sum_{i=1}^{k} \beta_i z_{i,t} + \varepsilon_t$$

where:
- $r_t$ is the log return at time $t$
- $\mu$ is the constant term
- $z_{i,t}$ are the PCA-transformed exogenous variables
- $\beta_i$ are the coefficients for the exogenous variables
- $\varepsilon_t$ is the error term following a Student's t-distribution

**Volatility Equation (EGARCH):**
$$\ln(\sigma_t^2) = \omega + \sum_{i=1}^{p} \alpha_i g(z_{t-i}) + \sum_{j=1}^{q} \beta_j \ln(\sigma_{t-j}^2)$$

where:
- $\sigma_t^2$ is the conditional variance at time $t$
- $\omega$, $\alpha_i$, and $\beta_j$ are parameters
- $g(z_t) = \theta z_t + \gamma[|z_t| - E|z_t|]$
- $z_t = \varepsilon_t/\sigma_t$ is the standardized residual
- $p$ and $q$ are the ARCH and GARCH orders respectively

The EGARCH model can capture asymmetric volatility effects, where negative returns impact volatility differently than positive returns of the same magnitude.

### 2.3 Dimension Reduction via PCA

Principal Component Analysis transforms the original exogenous variables into orthogonal components:

$$\mathbf{Z} = \mathbf{X} \cdot \mathbf{W}$$

where:
- $\mathbf{Z}$ is the matrix of principal components
- $\mathbf{X}$ is the original feature matrix
- $\mathbf{W}$ is the matrix of eigenvectors

The implementation retains at most 5 components, reducing multicollinearity while preserving most of the variance.

## 3. Implementation Logic

### 3.1 Rolling Forecast Origin

For each day $t$ with sufficient historical data:
1. A training window of length `forecast_lookback` (default: 200 days) is used: $[t-L, t-1]$
2. An EGARCH-X model is fitted to this training window
3. Forecasts are generated for the next `forecast_horizon` days (default: 30 days): $[t, t+H-1]$

### 3.2 Model Selection

For each training window, multiple EGARCH specifications are tested:
- Candidate orders: (1,1), (1,2), and (2,1) for $(p,q)$
- Each model is fitted with Student's t-distributed errors
- The model with the lowest AIC is selected:
  $$\text{AIC} = -2\ln(L) + 2k$$
  where $L$ is the likelihood and $k$ is the number of parameters

### 3.3 Forecasting

For the selected model:
1. Forecast mean returns for each day in the horizon:
   $$\hat{r}_{t+h} \text{ for } h=1,2,\ldots,H$$
2. Compute cumulative return forecast:
   $$\text{cumulative\_ret} = \exp\left(\sum_{h=1}^{H} \hat{r}_{t+h}\right) - 1$$
3. Compute forecast volatility:
   $$\sigma_{\text{forecast}} = \sqrt{\frac{1}{H}\sum_{h=1}^{H}\hat{\sigma}_{t+h}^2}$$

### 3.4 Signal Generation

The risk-adjusted return is calculated as:
$$\text{risk\_adj\_return} = \frac{\text{cumulative\_ret}}{\sigma_{\text{forecast}}}$$

Signal strength incorporates volume information:
$$\text{signal\_strength} = \text{risk\_adj\_return} \times \text{vol\_z}_t$$

The trading signal is determined by:
- Long (1): if $\text{risk\_adj\_return} > 0$ and $\text{vol\_z} > \text{min\_volume\_strength}$
- Short (-1): if $\text{risk\_adj\_return} < 0$ and $\text{vol\_z} > \text{min\_volume\_strength}$ (and not long-only)
- Neutral (0): otherwise

Position sizing is proportional to the risk-adjusted return:
$$\text{position\_size} = \text{risk\_adj\_return} \times \text{capital}$$

### 3.5 EWMA Fallback

If all EGARCH model fits fail, a simpler exponential weighted moving average (EWMA) model is used:
$$\hat{r}_t = \frac{\sum_{i=1}^{n} w_i r_{t-i}}{\sum_{i=1}^{n} w_i}$$

where $w_i = \lambda^{i-1}$ with $\lambda = 2/(30+1)$ for a 30-day span.

### 3.6 Risk Management

The generated signals are passed to a RiskManager that applies:
- Stop-loss: Exit if drawdown exceeds threshold
- Take-profit: Exit if profit exceeds threshold
- Slippage: Adjust entry/exit prices by a percentage
- Transaction costs: Deduct costs from returns

## 4. Advantages and Limitations

### Advantages
- Captures both mean and volatility dynamics
- Incorporates exogenous predictors via ARX specification
- Handles volatility clustering and asymmetric effects
- Provides risk-adjusted signals
- Avoids look-ahead bias through proper rolling window design
- Dynamically selects model orders

### Limitations
- Computationally intensive for large datasets
- Sensitive to outliers and noise in financial data
- Requires sufficient historical data for reliable parameter estimation
- May overfit in rapidly changing market regimes

## 5. Implementation Notes

- PCA transformation requires minimum 5 components to capture adequate variance
- Winsorization at 1% and 99% quantiles prevents outlier influence
- Multi-ticker processing is parallelized using joblib
- Model configurations are cached for future reference

---
# Mathematical Explanation of Prophet Momentum Strategy

The Prophet Momentum Strategy is a quantitative trading approach that combines time series forecasting with momentum detection to generate trading signals. This document provides a mathematical explanation of the key components of this strategy.

## 1. Prophet Time Series Forecasting

The strategy utilizes Facebook's Prophet, an additive time series forecasting model expressed as:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Where:
- $y(t)$ is the price forecast at time $t$
- $g(t)$ represents the trend component
- $s(t)$ represents seasonality components
- $h(t)$ represents holiday effects
- $\epsilon_t$ is the error term

The model is customized with ticker-specific hyperparameters optimized through forward cross-validation:
- Changepoint prior scale ($\delta_{cp}$): Controls flexibility of the trend
- Seasonality prior scale ($\delta_s$): Controls strength of seasonality
- Holidays prior scale ($\delta_h$): Controls impact of holidays
- Seasonality mode (additive or multiplicative)
- Fourier order ($k$): Controls complexity of seasonal patterns

## 2. Hyperparameter Optimization

Hyperparameters are optimized by maximizing directional accuracy through forward cross-validation. For each fold:

$$\text{DA} = \frac{1}{N} \sum_{i=1}^{N} I[\text{sign}(\hat{y}_i - y_{\text{last}}) = \text{sign}(y_i - y_{\text{last}})]$$

Where:
- $\hat{y}_i$ is the forecasted price
- $y_i$ is the actual price
- $y_{\text{last}}$ is the last observed price in the training set
- $I[·]$ is the indicator function

The optimization objective is to minimize loss:

$$\text{Loss} = 1 - \text{DA}$$

## 3. Forecast Confidence Analysis

The strategy analyzes confidence in the forecasted direction using multiple time horizons:
- Short horizon: $h_{\text{short}} = 7$ days
- Medium horizon: $h_{\text{mid}} = 14$ days
- Long horizon: $h_{\text{long}} = 30$ days

For each horizon, Prophet generates forecasts with confidence intervals:
- $\hat{y}_t$ (point forecast)
- $\hat{y}_t^{\text{lower}}$ (lower bound)
- $\hat{y}_t^{\text{upper}}$ (upper bound)

## 4. Trend Significance Testing

The strategy assesses the statistical significance of the forecasted trend using linear regression on the forecast points:

$$\hat{y}_t = \beta_0 + \beta_1 t + \varepsilon_t$$

The standard error of the slope ($\text{SE}_{\beta_1}$) is computed, and the t-statistic is calculated:

$$t = \frac{\beta_1}{\text{SE}_{\beta_1}}$$

This t-statistic measures the significance of the forecasted trend.

## 5. Market Regime Detection

A Hidden Markov Model (HMM) with Gaussian emissions is used to detect market regimes based on volatility:

$$\sigma_t = \text{StdDev}(r_{t-21:t})$$

Where $r_t$ represents daily returns. The HMM identifies two states:
- Low volatility state
- High volatility state

The model is specified as:

$$p(z_t|z_{t-1}) = A_{z_{t-1},z_t}$$
$$p(\sigma_t|z_t) = \mathcal{N}(\mu_{z_t}, \Sigma_{z_t})$$

Where:
- $z_t$ is the hidden state at time $t$
- $A$ is the transition matrix
- $\mathcal{N}(\mu_{z_t}, \Sigma_{z_t})$ is a Gaussian distribution with mean $\mu_{z_t}$ and variance $\Sigma_{z_t}$

## 6. Signal Generation

Trading signals are generated based on the following conditions:

For a long signal ($S_t = 1$):
- $\hat{y}_{t+h_{\text{short}}}^{\text{lower}} > P_t$ (Lower bound of short forecast > current price)
- $\hat{y}_{t+h_{\text{long}}}^{\text{upper}} > \hat{y}_{t+h_{\text{mid}}}^{\text{lower}}$ (Confidence intervals are monotonically rising)
- $t > 2$ (Slope is significantly positive)

For a short signal ($S_t = -1$):
- $\hat{y}_{t+h_{\text{short}}}^{\text{upper}} < P_t$ (Upper bound of short forecast < current price)
- $\hat{y}_{t+h_{\text{long}}}^{\text{lower}} < \hat{y}_{t+h_{\text{mid}}}^{\text{upper}}$ (Confidence intervals are monotonically falling)
- $t < -2$ (Slope is significantly negative)

Otherwise, $S_t = 0$ (no signal).

## 7. Signal Strength Calculation

Signal strength is calculated to determine position sizing:

For long positions:
$$\text{Strength} = \frac{\hat{y}_{t+h_{\text{long}}} - P_t}{P_t} \cdot \frac{t}{2}$$

For short positions:
$$\text{Strength} = \frac{P_t - \hat{y}_{t+h_{\text{long}}}}{P_t} \cdot \frac{|t|}{2}$$

## 8. Risk Management

Risk parameters include:
- Stop loss percentage: $\text{SL} = 5\%$
- Take profit percentage: $\text{TP} = 10\%$
- Slippage percentage: $\text{Slip} = 0.1\%$
- Transaction cost percentage: $\text{TC} = 0.1\%$

These are applied to manage position risk and account for trading frictions.

## 9. Market Regime Override

In high-volatility regimes (as detected by the HMM), all signals are neutralized:

$$\text{If } z_t = \text{high volatility}, \text{ then } S_t = 0 \text{ and Strength} = 0$$

This acts as a risk-off mechanism during turbulent market conditions.