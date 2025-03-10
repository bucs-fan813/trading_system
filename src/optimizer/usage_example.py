"""
Usage Demonstration
File: usage_example.py

This example demonstrates how to use the improved StrategyOptimizer and SensitivityAnalyzer.
"""

# Import the components from the optimizer package
from src.optimizer.strategy_optimizer import StrategyOptimizer
from src.optimizer.sensitivity_analyzer import SensitivityAnalyzer
from src.strategies.momentum.awesome_oscillator_strat import AwesomeOscillatorStrategy
from hyperopt import hp

# Example configuration
db_config = {"host": "localhost", "port": 1234}
example_search_space = {
    'window_size': hp.quniform('window_size', 5, 30, 1),
    'threshold': hp.uniform('threshold', 0.0, 1.0)
}

# Define tickers and optional weights
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META']
ticker_weights = {
    'AAPL': 0.3,
    'GOOG': 0.2,
    'MSFT': 0.2,
    'AMZN': 0.2,
    'META': 0.1
}

# Initialize and run the optimizer with multiple tickers and weights
optimizer = StrategyOptimizer(
    strategy_class=AwesomeOscillatorStrategy,  # Your strategy class
    db_config=db_config,
    search_space=example_search_space,
    tickers=tickers,
    ticker_weights=ticker_weights,  # Optional: provide custom weights
    start_date="2020-01-01",
    end_date="2022-12-31",
    max_evals=50
)

# Run optimization
best_params, performance_report, param_history = optimizer.run_optimization()
print("Best Parameters:")
print(best_params)
print("\nPerformance Report:")
print(performance_report)

# Run sensitivity analysis (reusing the optimizer instance for efficiency)
sensitivity_analyzer = SensitivityAnalyzer(
    strategy_optimizer=optimizer,  # Pass the optimizer instance directly
    base_params=best_params,
    num_samples=30
)

sensitivity_results, parameter_impact = sensitivity_analyzer.run()
print("\nParameter Impact Analysis:")
print(parameter_impact)