# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an **iPhone Demand Forecasting System** for S&OP (Sales & Operations Planning) decision support. The repository contains machine learning-based demand forecasting models with scenario simulation capabilities for production planning, inventory optimization, and promotional investment allocation.

## Technology Stack

- **Language**: Python 3.12+
- **ML Framework**: LightGBM (Gradient Boosting)
- **Data Processing**: pandas, numpy
- **Model Evaluation**: scikit-learn
- **Output**: openpyxl (Excel exports)

## Project Structure

```
/workspaces/Operation-Management/
├── requirements_document.md          # Comprehensive business requirements
├── README.md                         # User documentation
├── CLAUDE.md                         # This file
├── Iphone_Sales_Data(1).csv         # Historical sales data (2020-2025)
├── iphone_forecast_refined.py       # Refined ML forecasting model v3.0 (MAPE 13.34%)
└── iphone_forecast/
    └── output/                       # Forecast outputs (CSV, PNG)
        ├── forecast_iphone16_refined.csv
        ├── forecast_iphone17_refined.csv
        └── forecast_refined_comparison.png
```

## Key Commands

### Run Forecasting Model

```bash
# Refined ML-based forecast with optimized hyperparameters (v3.0)
python3 iphone_forecast_refined.py
```

### Install Dependencies

```bash
pip install pandas numpy lightgbm scikit-learn matplotlib optuna
```

## Code Architecture

### Refined Forecasting Model (`iphone_forecast_refined.py`) v3.0

**Purpose**: Production-ready ML forecasting with optimized hyperparameters and outlier handling

**Architecture**:

```
[Data Loading] → [Outlier Detection] → [Feature Engineering] → [Hyperparameter Optimization] → [Model Training] → [Evaluation] → [Forecasting] → [Export]
     ↓                  ↓                       ↓                          ↓                         ↓                 ↓              ↓             ↓
  2,490 records    ±2σ smoothing          24 features                 Optuna (20 trials)         LightGBM        MAPE 13.34%    iPhone 16     CSV/PNG
  5 models         4 outliers             - Log-transformed lags      - num_leaves: 101          Optimized       WAPE 13.56%    iPhone 17     3 versions
  2020-2025        corrected              - Growth metrics            - learning_rate: 0.0229    Time-series CV  Target ✓       Comparison
                                          - Acceleration              - L1/L2 regularization
                                          - Interaction terms
```

#### Feature Engineering (24 features)

**Core Features** (18 original):
- Temporal: Month, Quarter, DayOfWeek, WeekOfYear, DaysSinceLaunch, WeeksSinceLaunch, MonthsSinceLaunch
- Lifecycle: LifecycleStage (Launch/Growth/Mature/Decline), IsHolidaySeason, IsBackToSchool, IsQ4
- Lags/MA: Sales_MA7, Sales_MA30, Sales_Lag7, Sales_Lag30
- Business: SimulatedPrice, PromoIntensity, IsNewModelLaunchMonth, ModelNumber

**New Features in v3.0** (6 additional):
- **Log-transformed lags**: lag1_log, lag2_log, lag3_log (stabilizes launch phase volatility)
- **Growth metrics**: mom_growth_ma3 (month-over-month smoothed growth), acceleration (growth rate change)
- **Interaction term**: launch_x_stage (captures launch lifecycle dynamics)

#### Model Performance Benchmarks

**Version Comparison** (iPhone 16, first 4 months):

| Version | MAPE (%) | WAPE (%) | Key Improvements |
|---------|----------|----------|------------------|
| v1.0 Baseline | 23.41% | 23.76% | Basic feature engineering |
| v2.0 Improved | 22.53% | 22.94% | Enhanced features (18 total) |
| v3.0 Refined | 13.34% ✓ | 13.56% ✓ | Outlier handling + Optuna + log lags |

**Target Achievement**:
- **MAPE ≤ 15%** (non-launch months): ✓ Achieved 13.34%
- **Cross-validation**: 3-fold time series CV, Best MAE: 0.784

**Top Feature Importance** (v3.0):
1. ema3 (exponential moving average): 15,435
2. ma3 (3-month moving average): 8,363
3. launch_month: 4,399

## Development Guidelines

### Adding New Features

1. **Data Features**: Add to feature engineering section in `iphone_forecast_refined.py`
2. **Engineering**: Implement after outlier detection, before model training
3. **Validation**: Check feature importance in output graphs

### Model Tuning

**Optimized LightGBM parameters** (via Optuna):
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 101,              # Optimized complexity
    'learning_rate': 0.0229,        # Optimized step size
    'feature_fraction': 0.765,      # Feature sampling
    'bagging_fraction': 0.734,      # Row sampling
    'bagging_freq': 5,
    'lambda_l1': 0.208,             # L1 regularization
    'lambda_l2': 0.313,             # L2 regularization
    'min_child_samples': 20,
    'verbose': -1
}
```

To re-optimize hyperparameters, modify Optuna trials in the code:
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Increase from 20 to 50
```

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error (symmetric, % based)
- **WAPE**: Weighted Absolute Percentage Error (volume-weighted)
- **MAE**: Mean Absolute Error (millions of units)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **Bias**: Average prediction error (over/under forecasting)

### Output Files

**Refined Model Output** (iphone_forecast/output/):
1. **forecast_iphone16_refined.csv**: iPhone 16 forecast comparison (v1.0, v2.0, v3.0)
2. **forecast_iphone17_refined.csv**: iPhone 17 forecast (Sept 2025 launch, 4 months)
3. **forecast_refined_comparison.png**: Visual comparison graph (3 versions)

## Common Development Tasks

### Update Model with New Data

1. Replace `Iphone_Sales_Data(1).csv` with updated data
2. Verify date format: `M/D/YYYY` (e.g., `10/1/2020`)
3. Ensure columns: `Date`, `Model`, `Estimated_Units_Millions`
4. Run: `python3 iphone_forecast_refined.py`

### Add New iPhone Model

1. Update `model_launch_dates` dict with new model and launch date
2. Model will automatically train on historical data and forecast new model
3. Output will include new model forecast in CSV and comparison graphs

## Business Context

### Success Criteria

- **MAPE ≤ 15%** for non-launch months (achieved: 13.34% ✓)
- **Demand stockout reduction**: Target -30%

### Stakeholders

- **Owner**: Ops (Demand Planning/Supply Planning)
- **Users**: Sales, Marketing, Finance, Supply Chain
- **Cadence**: Weekly Ops meetings, Monthly S&OP

### Use Cases

1. **S&OP Planning**: Monthly demand plan by model/channel/country
2. **Production Planning**: M+3 to M+6 procurement
3. **Inventory Optimization**: Launch planning, EOL clearance
4. **Promotional Planning**: Price/promo impact simulation

## Data Schema

### Input Data (CSV)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Date | datetime | Sales date | 10/1/2020 |
| Model | string | iPhone model | iPhone 12 |
| Estimated_Units_Millions | float | Daily units (millions) | 0.751455 |

### Future Extensions (from requirements_document.md)

**Phase 2 - Data Enrichment**:
- Price data (realized price, discount rate)
- Promotion flags (campaign types, intensity)
- Inventory data (on-hand days, stockout flags)
- Channel segmentation (carrier/retail/online)
- Geographic hierarchy (country/region)

**Phase 3 - Model Enhancement**:
- Hierarchical forecasting (top-down/bottom-up reconciliation)
- Ensemble methods (Prophet + LightGBM)
- Bayesian cold start (pre-order signals)
- Demand uncensoring (stockout correction)

**Phase 4 - MLOps**:
- Automated retraining (weekly/monthly)
- Dashboard (interactive pivots)
- REST API endpoints
- Drift monitoring (data + error)

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'lightgbm'`
**Fix**: `pip install lightgbm scikit-learn`

**Issue**: MAPE very high (>20%)
**Check**:
- Data quality (missing values, outliers)
- Feature engineering (are lags/MA populated?)
- Model parameters (increase `num_leaves`, decrease `learning_rate`)

**Issue**: Forecast seems unrealistic
**Check**:
- Launch dates in `model_launch_dates` are correct
- SimulatedPrice calculation aligns with actual pricing
- PromoIntensity flags match actual promotional calendar

## Performance Optimization

Current model trains in <30 seconds on 2,490 records.

For larger datasets:
- Use `bagging_freq` and `feature_fraction` for faster training
- Reduce `num_boost_round` (currently 500)
- Consider data sampling for initial experiments

## References

- **Business Requirements**: `requirements_document.md`
- **User Guide**: `README.md`
- **LightGBM Docs**: https://lightgbm.readthedocs.io/
- **Optuna Docs**: https://optuna.readthedocs.io/
- **Time Series CV**: scikit-learn TimeSeriesSplit

## Contact

For questions about:
- **Model methodology**: See `requirements_document.md`
- **Business requirements**: See `requirements_document.md`
- **Technical implementation**: Review `iphone_forecast_refined.py` inline comments
