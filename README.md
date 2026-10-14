# Multi-Timescale Volatility Forecasting

A continuous-time volatility model that represents SPY volatility memory as a mixture of Ornstein–Uhlenbeck (OU) processes.

## Approach

Using ~24 years of high-frequency SPY data, I estimate the empirical covariance structure of realised volatility and find strong evidence for multiple persistence timescales.

A two-factor OU model identifies:
- **Fast component:** ~3.6 effective hours
- **Slow component:** ~497 effective hours
- **82% lower covariance-fit MSE** than a Single-OU benchmark

The latent factors are estimated using a Kalman filter with exact continuous-time OU transitions. An open/closed-market clock allows volatility to evolve at a different rate outside trading hours.

## Out-of-Sample Results

Models are trained pre-2017, the market clock is selected on 2017–2018, and final performance is evaluated on an untouched post-2019 test set.

| Horizon | Multi-OU log-MSE reduction |
| ---: | ---: |
| 1 day | 7.0% |
| 7 days | **12.6%** |
| 30 days | 5.6% |
| 90 days | 2.0% |

QLIKE favours Multi-OU at 1- and 7-day horizons, while Single OU performs marginally better at 30 and 90 days.

## Repository

- `01_memory_analysis.ipynb` — empirical memory and Bernstein representation
- `02_model_calibration.ipynb` — Multi-OU calibration
- `03_forecasting.ipynb` — out-of-sample forecasting
- `implementation/` — reusable modelling and backtesting code