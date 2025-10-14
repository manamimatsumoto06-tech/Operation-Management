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
├── iphone16_forecast.py             # Baseline forecasting model
├── advanced_forecast_model.py       # Advanced ML-based forecasting (RECOMMENDED)
├── iPhone16_Sales_Forecast.xlsx     # Baseline output
└── iPhone_Advanced_Forecast_Output.xlsx  # Advanced model output
```

## Key Commands

### Run Forecasting Models

```bash
# Baseline time series forecast (simple)
python3 iphone16_forecast.py

# Advanced ML-based forecast with scenarios (recommended)
python3 advanced_forecast_model.py
```

### Install Dependencies

```bash
# Baseline model
pip install pandas numpy openpyxl

# Advanced model
pip install pandas numpy openpyxl lightgbm scikit-learn
```

## Code Architecture

### 1. Baseline Model (`iphone16_forecast.py`)

**Purpose**: Simple time series forecasting using historical patterns
**Approach**:
- Aggregate daily sales to monthly
- Calculate average sales pattern across iPhone 12-15
- Apply 3% growth rate to iPhone 15 baseline
- Export 4-sheet Excel report

**Use case**: Quick estimates, initial planning

### 2. Advanced Model (`advanced_forecast_model.py`) ⭐ RECOMMENDED

**Purpose**: Enterprise-grade ML forecasting with scenario simulation
**Architecture**:

```
[Data Loading] → [Feature Engineering] → [Model Training] → [Evaluation] → [Forecasting] → [Scenarios] → [Export]
     ↓                  ↓                      ↓                  ↓              ↓             ↓            ↓
  2,490 records    18 features          LightGBM           MAPE/WAPE      iPhone 16     3 scenarios    6 sheets
  5 models         - Lifecycle          500 trees          1.94%          12 months     Price/Promo   Excel
  2020-2025        - Seasonality        Time-series CV     1.70% WAPE     Daily→Monthly Supply
                   - Price
                   - Promo
                   - Lags/MA
```

#### Feature Engineering (18 features)

**Temporal Features**:
- Month, Quarter, DayOfWeek, WeekOfYear
- DaysSinceLaunch, WeeksSinceLaunch, MonthsSinceLaunch

**Lifecycle Features**:
- LifecycleStage: Launch/Growth/Mature/Decline
- IsHolidaySeason (Nov-Dec)
- IsBackToSchool (Aug-Sep)
- IsQ4 (Q4 flag)

**Demand Signals**:
- Sales_MA7, Sales_MA30 (moving averages)
- Sales_Lag7, Sales_Lag30 (lag features)

**Business Factors**:
- SimulatedPrice (lifecycle-based pricing)
- PromoIntensity (Black Friday, holiday promos)
- IsNewModelLaunchMonth (cannibalization)
- ModelNumber (generation encoding)

#### Model Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| MAPE (%) | ≤12% | 1.94% ✓ |
| WAPE (%) | ≤18% (launch) | 1.70% ✓ |
| iPhone 15 MAPE | - | 1.88% |

#### Scenario Simulation

**Price -5% Scenario**:
- Elasticity: -1.2 (elastic demand)
- Impact: +6.0% demand
- Use: Price optimization, margin analysis

**Promo +1 Level Scenario**:
- Lift: +8% demand
- Use: Promotional ROI analysis

**Supply -20% Scenario**:
- Constraint: 80% fulfillment
- Impact: -20% sales (lost sales)
- Use: Inventory risk analysis

## Development Guidelines

### Adding New Features

1. **Data Features**: Add to `feature_cols` list in `advanced_forecast_model.py`
2. **Engineering**: Implement in section [2/7] Feature Engineering
3. **Validation**: Check feature importance in output

### Model Tuning

LightGBM parameters in `params` dict:
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,          # Complexity (increase for more complex patterns)
    'learning_rate': 0.05,     # Step size (lower = more stable)
    'feature_fraction': 0.8,   # Feature sampling
    'bagging_fraction': 0.8,   # Row sampling
    'bagging_freq': 5,         # Bagging frequency
}
```

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error (symmetric, % based)
- **WAPE**: Weighted Absolute Percentage Error (volume-weighted)
- **MAE**: Mean Absolute Error (millions of units)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **Bias**: Average prediction error (over/under forecasting)

### Output Files

**Advanced Model Excel Output** (6 sheets):
1. `Monthly_Forecast_Scenarios`: 12-month forecast + 3 scenarios
2. `Model_Performance`: MAPE/WAPE by segment (model, lifecycle)
3. `Feature_Importance`: Gain-based feature ranking
4. `Daily_Forecast_Detail`: First 90 days breakdown
5. `Historical_Validation`: Actuals vs Predicted (training period)
6. `Executive_Summary`: Key metrics dashboard

## Common Development Tasks

### Update Model with New Data

1. Replace `Iphone_Sales_Data(1).csv` with updated data
2. Verify date format: `M/D/YYYY` (e.g., `10/1/2020`)
3. Ensure columns: `Date`, `Model`, `Estimated_Units_Millions`
4. Run: `python3 advanced_forecast_model.py`

### Add New iPhone Model

1. Update `model_launch_dates` dict with new model and launch date
2. Update `model_number` dict with model encoding
3. Model will automatically train on historical data and forecast new model

### Customize Scenarios

In section [6/7], modify scenario parameters:
```python
# Price scenario
scenario1['SimulatedPrice'] = scenario1['SimulatedPrice'] * 0.90  # -10% instead of -5%
price_elasticity = -1.5  # More elastic

# Promo scenario
scenario2['PromoIntensity'] = scenario2['PromoIntensity'] + 2.0  # +2 levels
promo_lift = 0.15  # 15% lift instead of 8%
```

### Export to Different Formats

Current: Excel (openpyxl)
To add CSV export:
```python
monthly_df.to_csv('forecast_output.csv', index=False)
```

## Business Context

### Success Criteria

- **MAPE ≤ 12%** for non-launch months (achieved: 1.94%)
- **WAPE ≤ 18%** for launch ±2 months (achieved: 1.70%)
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
- **Time Series CV**: scikit-learn TimeSeriesSplit

## Contact

For questions about:
- **Model methodology**: See `requirements_document.md` section 7
- **Business requirements**: See `requirements_document.md` section 1-5
- **Technical implementation**: Review `advanced_forecast_model.py` inline comments
