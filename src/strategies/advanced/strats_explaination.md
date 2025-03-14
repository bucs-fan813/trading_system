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