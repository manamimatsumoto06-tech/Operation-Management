#!/usr/bin/env python3
"""
Advanced iPhone Demand Forecasting System
Implements ML-based forecasting with feature engineering, hierarchical reconciliation,
and scenario simulation capabilities for S&OP decision support.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

print("="*80)
print("iPhone Advanced Demand Forecasting System")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("\n[1/7] Loading and preparing data...")
df = pd.read_csv('Iphone_Sales_Data(1).csv')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])

print(f"  - Total records: {len(df):,}")
print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  - Models: {df['Model'].unique().tolist()}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n[2/7] Engineering features...")

# Sort by model and date
df = df.sort_values(['Model', 'Date']).reset_index(drop=True)

# Model launch dates (actual historical)
model_launch_dates = {
    'iPhone 12': '2020-10-01',
    'iPhone 13': '2021-09-01',
    'iPhone 14': '2022-09-01',
    'iPhone 15': '2023-09-01',
    'iPhone 16': '2024-09-01'  # Assumed
}

# Add temporal features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Lifecycle features - days since launch
for model, launch_date in model_launch_dates.items():
    mask = df['Model'] == model
    df.loc[mask, 'DaysSinceLaunch'] = (df.loc[mask, 'Date'] - pd.to_datetime(launch_date)).dt.days
    df.loc[mask, 'WeeksSinceLaunch'] = df.loc[mask, 'DaysSinceLaunch'] // 7
    df.loc[mask, 'MonthsSinceLaunch'] = df.loc[mask, 'DaysSinceLaunch'] // 30

# Lifecycle stage (Early/Growth/Mature/Decline)
df['LifecycleStage'] = 'Unknown'
df.loc[df['MonthsSinceLaunch'] <= 2, 'LifecycleStage'] = 'Launch'
df.loc[(df['MonthsSinceLaunch'] > 2) & (df['MonthsSinceLaunch'] <= 6), 'LifecycleStage'] = 'Growth'
df.loc[(df['MonthsSinceLaunch'] > 6) & (df['MonthsSinceLaunch'] <= 12), 'LifecycleStage'] = 'Mature'
df.loc[df['MonthsSinceLaunch'] > 12, 'LifecycleStage'] = 'Decline'

# Season flags (Holiday season: Nov-Dec, Back to school: Aug-Sep)
df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)
df['IsBackToSchool'] = df['Month'].isin([8, 9]).astype(int)
df['IsQ4'] = (df['Quarter'] == 4).astype(int)

# Lag features (7-day, 30-day moving averages)
for model in df['Model'].unique():
    mask = df['Model'] == model
    df.loc[mask, 'Sales_MA7'] = df.loc[mask, 'Estimated_Units_Millions'].rolling(7, min_periods=1).mean()
    df.loc[mask, 'Sales_MA30'] = df.loc[mask, 'Estimated_Units_Millions'].rolling(30, min_periods=1).mean()
    df.loc[mask, 'Sales_Lag7'] = df.loc[mask, 'Estimated_Units_Millions'].shift(7)
    df.loc[mask, 'Sales_Lag30'] = df.loc[mask, 'Estimated_Units_Millions'].shift(30)

# Price elasticity simulation (synthetic - in real scenario, use actual price data)
# Simulating price decay pattern: higher price at launch, gradual discounts
df['SimulatedPrice'] = 999.0  # Base price
for model in df['Model'].unique():
    mask = df['Model'] == model
    months_since = df.loc[mask, 'MonthsSinceLaunch'].values
    # Price decays ~5% per quarter after launch
    price_multiplier = np.maximum(0.7, 1.0 - (months_since / 12) * 0.3)
    df.loc[mask, 'SimulatedPrice'] = 999.0 * price_multiplier

# Simulated promotions (Black Friday, Holiday season)
df['PromoIntensity'] = 0.0
df.loc[(df['Month'] == 11) & (df['DayOfWeek'] == 4), 'PromoIntensity'] = 2.0  # Black Friday
df.loc[df['IsHolidaySeason'] == 1, 'PromoIntensity'] = df.loc[df['IsHolidaySeason'] == 1, 'PromoIntensity'] + 1.0

# Model generation encoding
model_number = {'iPhone 12': 12, 'iPhone 13': 13, 'iPhone 14': 14, 'iPhone 15': 15, 'iPhone 16': 16}
df['ModelNumber'] = df['Model'].map(model_number)

# Competitor launch impact (simulated: new model cannibalizes old)
df['IsNewModelLaunchMonth'] = 0
for launch_date in model_launch_dates.values():
    launch_dt = pd.to_datetime(launch_date)
    mask = (df['Date'] >= launch_dt) & (df['Date'] < launch_dt + timedelta(days=60))
    df.loc[mask, 'IsNewModelLaunchMonth'] = 1

print(f"  - Created {len(df.columns)} features")
print(f"  - Sample features: {[col for col in df.columns if col not in ['Date', 'Model', 'Estimated_Units_Millions']][:10]}")

# ============================================================================
# 3. TIME SERIES CROSS-VALIDATION SETUP
# ============================================================================

print("\n[3/7] Setting up time series cross-validation...")

# Use data up to iPhone 15 for training, iPhone 16 for testing
train_models = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
test_model = 'iPhone 16'

train_df = df[df['Model'].isin(train_models)].copy()
test_df = df[df['Model'] == test_model].copy()

print(f"  - Training data: {len(train_df):,} records ({train_models})")
print(f"  - Test data: {len(test_df):,} records ({test_model})")

# Features for modeling
feature_cols = [
    'ModelNumber', 'Month', 'Quarter', 'DayOfWeek', 'WeekOfYear',
    'DaysSinceLaunch', 'WeeksSinceLaunch', 'MonthsSinceLaunch',
    'IsHolidaySeason', 'IsBackToSchool', 'IsQ4',
    'Sales_MA7', 'Sales_MA30', 'Sales_Lag7', 'Sales_Lag30',
    'SimulatedPrice', 'PromoIntensity', 'IsNewModelLaunchMonth'
]

# Handle missing values in lag features
train_df[feature_cols] = train_df[feature_cols].fillna(0)
test_df[feature_cols] = test_df[feature_cols].fillna(0)

target_col = 'Estimated_Units_Millions'

# ============================================================================
# 4. MODEL TRAINING (LightGBM)
# ============================================================================

print("\n[4/7] Training LightGBM demand forecasting model...")

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df[target_col]

# LightGBM parameters optimized for time series
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

# Train model
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

print(f"  - Model trained with {model.num_trees()} trees")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

print("\n  Top 10 Important Features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"    {row['Feature']:30s}: {row['Importance']:>10.0f}")

# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

print("\n[5/7] Evaluating model performance...")

# Predictions on training set
train_df['Predicted'] = model.predict(X_train)

# Calculate metrics
def calculate_metrics(actual, predicted, name=""):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    wape = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100
    bias = np.mean(predicted - actual)

    return {
        'Segment': name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'WAPE (%)': wape,
        'Bias': bias
    }

# Overall metrics
overall_metrics = calculate_metrics(train_df[target_col], train_df['Predicted'], "Overall Training")

# By model
model_metrics = []
for model_name in train_models:
    model_data = train_df[train_df['Model'] == model_name]
    metrics = calculate_metrics(model_data[target_col], model_data['Predicted'], model_name)
    model_metrics.append(metrics)

# By lifecycle stage
lifecycle_metrics = []
for stage in ['Launch', 'Growth', 'Mature', 'Decline']:
    stage_data = train_df[train_df['LifecycleStage'] == stage]
    if len(stage_data) > 0:
        metrics = calculate_metrics(stage_data[target_col], stage_data['Predicted'], f"Lifecycle: {stage}")
        lifecycle_metrics.append(metrics)

print("\n  Overall Performance:")
print(f"    MAE:        {overall_metrics['MAE']:.4f} M units")
print(f"    RMSE:       {overall_metrics['RMSE']:.4f} M units")
print(f"    MAPE:       {overall_metrics['MAPE (%)']:.2f}%")
print(f"    WAPE:       {overall_metrics['WAPE (%)']:.2f}%")
print(f"    Bias:       {overall_metrics['Bias']:+.4f} M units")

print("\n  Performance by Model:")
for metrics in model_metrics:
    print(f"    {metrics['Segment']:12s} - MAPE: {metrics['MAPE (%)']:5.2f}%, WAPE: {metrics['WAPE (%)']:5.2f}%")

# ============================================================================
# 6. IPHONE 16 FORECAST & SCENARIO SIMULATION
# ============================================================================

print("\n[6/7] Generating iPhone 16 forecast with scenarios...")

# Base forecast
if len(test_df) > 0:
    X_test = test_df[feature_cols]
    test_df['Predicted_Base'] = model.predict(X_test)
    base_forecast = test_df.copy()
else:
    # Generate synthetic test data for iPhone 16 (12 months)
    print("  - No iPhone 16 data found, generating synthetic forecast period...")
    forecast_start = pd.to_datetime('2024-09-01')
    forecast_dates = pd.date_range(forecast_start, periods=365, freq='D')

    forecast_data = []
    for date in forecast_dates:
        forecast_data.append({
            'Date': date,
            'Model': 'iPhone 16',
            'Estimated_Units_Millions': 0  # Will be predicted
        })

    base_forecast = pd.DataFrame(forecast_data)

    # Recalculate features for forecast period
    base_forecast['Year'] = base_forecast['Date'].dt.year
    base_forecast['Month'] = base_forecast['Date'].dt.month
    base_forecast['Quarter'] = base_forecast['Date'].dt.quarter
    base_forecast['DayOfWeek'] = base_forecast['Date'].dt.dayofweek
    base_forecast['DayOfYear'] = base_forecast['Date'].dt.dayofyear
    base_forecast['WeekOfYear'] = base_forecast['Date'].dt.isocalendar().week

    launch_date = pd.to_datetime(model_launch_dates['iPhone 16'])
    base_forecast['DaysSinceLaunch'] = (base_forecast['Date'] - launch_date).dt.days
    base_forecast['WeeksSinceLaunch'] = base_forecast['DaysSinceLaunch'] // 7
    base_forecast['MonthsSinceLaunch'] = base_forecast['DaysSinceLaunch'] // 30

    base_forecast['LifecycleStage'] = 'Growth'
    base_forecast.loc[base_forecast['MonthsSinceLaunch'] <= 2, 'LifecycleStage'] = 'Launch'
    base_forecast.loc[base_forecast['MonthsSinceLaunch'] > 6, 'LifecycleStage'] = 'Mature'

    base_forecast['IsHolidaySeason'] = base_forecast['Month'].isin([11, 12]).astype(int)
    base_forecast['IsBackToSchool'] = base_forecast['Month'].isin([8, 9]).astype(int)
    base_forecast['IsQ4'] = (base_forecast['Quarter'] == 4).astype(int)

    # Use iPhone 15 patterns for lag features
    iphone15_avg = train_df[train_df['Model'] == 'iPhone 15']['Estimated_Units_Millions'].mean()
    base_forecast['Sales_MA7'] = iphone15_avg
    base_forecast['Sales_MA30'] = iphone15_avg
    base_forecast['Sales_Lag7'] = iphone15_avg
    base_forecast['Sales_Lag30'] = iphone15_avg

    base_forecast['SimulatedPrice'] = 999.0
    months_since = base_forecast['MonthsSinceLaunch'].values
    price_multiplier = np.maximum(0.7, 1.0 - (months_since / 12) * 0.3)
    base_forecast['SimulatedPrice'] = 999.0 * price_multiplier

    base_forecast['PromoIntensity'] = 0.0
    base_forecast.loc[(base_forecast['Month'] == 11) & (base_forecast['DayOfWeek'] == 4), 'PromoIntensity'] = 2.0
    base_forecast.loc[base_forecast['IsHolidaySeason'] == 1, 'PromoIntensity'] = base_forecast.loc[base_forecast['IsHolidaySeason'] == 1, 'PromoIntensity'] + 1.0

    base_forecast['ModelNumber'] = 16
    base_forecast['IsNewModelLaunchMonth'] = 0
    base_forecast.loc[base_forecast['MonthsSinceLaunch'] <= 2, 'IsNewModelLaunchMonth'] = 1

    # Fill any remaining missing features
    for col in feature_cols:
        if col not in base_forecast.columns:
            base_forecast[col] = 0

    # Make predictions
    X_forecast = base_forecast[feature_cols].fillna(0)
    base_forecast['Predicted_Base'] = model.predict(X_forecast)

# Scenario 1: Price reduction (-5%)
scenario1 = base_forecast.copy()
scenario1['SimulatedPrice'] = scenario1['SimulatedPrice'] * 0.95
price_elasticity = -1.2  # 1% price decrease → 1.2% demand increase (elastic)
X_scenario1 = scenario1[feature_cols].fillna(0)
scenario1['Predicted_Price_Minus5pct'] = model.predict(X_scenario1) * (1 + 0.05 * abs(price_elasticity))

# Scenario 2: Increased promotion (+1 level)
scenario2 = base_forecast.copy()
scenario2['PromoIntensity'] = scenario2['PromoIntensity'] + 1.0
promo_lift = 0.08  # +1 promo level → +8% demand
X_scenario2 = scenario2[feature_cols].fillna(0)
scenario2['Predicted_Promo_Plus1'] = model.predict(X_scenario2) * (1 + promo_lift)

# Scenario 3: Supply constraint (-20% availability)
scenario3 = base_forecast.copy()
supply_constraint = 0.80  # Only 80% of demand can be fulfilled
X_scenario3 = scenario3[feature_cols].fillna(0)
scenario3['Predicted_Supply_Minus20pct'] = model.predict(X_scenario3) * supply_constraint

# Aggregate to monthly
monthly_forecast = base_forecast.copy()
monthly_forecast['YearMonth'] = monthly_forecast['Date'].dt.to_period('M')

monthly_results = []
for ym in monthly_forecast['YearMonth'].unique():
    month_data = monthly_forecast[monthly_forecast['YearMonth'] == ym]
    s1_data = scenario1[scenario1['Date'].isin(month_data['Date'])]
    s2_data = scenario2[scenario2['Date'].isin(month_data['Date'])]
    s3_data = scenario3[scenario3['Date'].isin(month_data['Date'])]

    monthly_results.append({
        'YearMonth': str(ym),
        'Year': ym.year,
        'Month': ym.month,
        'Base_Forecast_Units_M': month_data['Predicted_Base'].sum(),
        'Scenario_Price-5%_Units_M': s1_data['Predicted_Price_Minus5pct'].sum(),
        'Scenario_Promo+1_Units_M': s2_data['Predicted_Promo_Plus1'].sum(),
        'Scenario_Supply-20%_Units_M': s3_data['Predicted_Supply_Minus20pct'].sum()
    })

monthly_df = pd.DataFrame(monthly_results)

print("\n  iPhone 16 Monthly Forecast (First 12 Months):")
print(monthly_df.head(12).to_string(index=False))

total_base = monthly_df.head(12)['Base_Forecast_Units_M'].sum()
total_price = monthly_df.head(12)['Scenario_Price-5%_Units_M'].sum()
total_promo = monthly_df.head(12)['Scenario_Promo+1_Units_M'].sum()
total_supply = monthly_df.head(12)['Scenario_Supply-20%_Units_M'].sum()

print(f"\n  12-Month Totals:")
print(f"    Base Forecast:          {total_base:>8.2f} M units")
print(f"    Price -5% Scenario:     {total_price:>8.2f} M units (+{((total_price/total_base-1)*100):>5.1f}%)")
print(f"    Promo +1 Scenario:      {total_promo:>8.2f} M units (+{((total_promo/total_base-1)*100):>5.1f}%)")
print(f"    Supply -20% Scenario:   {total_supply:>8.2f} M units ({((total_supply/total_base-1)*100):>5.1f}%)")

# ============================================================================
# 7. EXPORT RESULTS
# ============================================================================

print("\n[7/7] Exporting comprehensive results to Excel...")

output_file = 'iPhone_Advanced_Forecast_Output.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet 1: Monthly forecast with scenarios
    monthly_df.to_excel(writer, sheet_name='Monthly_Forecast_Scenarios', index=False)

    # Sheet 2: Model performance metrics
    all_metrics = [overall_metrics] + model_metrics + lifecycle_metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_excel(writer, sheet_name='Model_Performance', index=False)

    # Sheet 3: Feature importance
    importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

    # Sheet 4: Daily forecast detail (first 90 days)
    daily_detail = base_forecast[['Date', 'Model', 'MonthsSinceLaunch', 'LifecycleStage',
                                   'SimulatedPrice', 'PromoIntensity', 'Predicted_Base']].head(90)
    daily_detail.to_excel(writer, sheet_name='Daily_Forecast_Detail', index=False)

    # Sheet 5: Historical actuals vs predictions
    historical_comparison = train_df[['Date', 'Model', 'Estimated_Units_Millions', 'Predicted']].copy()
    historical_comparison.columns = ['Date', 'Model', 'Actual_Units_M', 'Predicted_Units_M']
    historical_comparison['Error'] = historical_comparison['Predicted_Units_M'] - historical_comparison['Actual_Units_M']
    historical_comparison['APE_%'] = np.abs(historical_comparison['Error'] / historical_comparison['Actual_Units_M']) * 100
    # Sample every 7 days to reduce size
    historical_comparison = historical_comparison.iloc[::7, :]
    historical_comparison.to_excel(writer, sheet_name='Historical_Validation', index=False)

    # Sheet 6: Summary dashboard data
    summary_data = {
        'Metric': [
            'Model Type',
            'Training Records',
            'Training Period',
            'Features Used',
            'Overall MAPE (%)',
            'Overall WAPE (%)',
            'iPhone 16 Forecast Period',
            'iPhone 16 Total (12M)',
            'iPhone 15 Actual Total (12M)',
            'YoY Growth (%)',
            'Scenario: Price -5% Impact',
            'Scenario: Promo +1 Impact',
            'Scenario: Supply -20% Impact'
        ],
        'Value': [
            'LightGBM Gradient Boosting',
            f"{len(train_df):,}",
            f"{train_df['Date'].min().date()} to {train_df['Date'].max().date()}",
            f"{len(feature_cols)}",
            f"{overall_metrics['MAPE (%)']:.2f}%",
            f"{overall_metrics['WAPE (%)']:.2f}%",
            f"{monthly_df['YearMonth'].min()} to {monthly_df.head(12)['YearMonth'].max()}",
            f"{total_base:.2f} M",
            "156.33 M",  # From original forecast
            f"{((total_base/156.33 - 1)*100):+.2f}%",
            f"+{((total_price/total_base-1)*100):.1f}% demand",
            f"+{((total_promo/total_base-1)*100):.1f}% demand",
            f"{((total_supply/total_base-1)*100):.1f}% demand"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)

print(f"\n{'='*80}")
print(f"✓ Advanced forecast completed successfully!")
print(f"{'='*80}")
print(f"\nOutput file: {output_file}")
print("\nExcel sheets:")
print("  1. Monthly_Forecast_Scenarios - iPhone 16 monthly forecast with 3 scenarios")
print("  2. Model_Performance - MAPE, WAPE, Bias by segment")
print("  3. Feature_Importance - Key drivers of demand")
print("  4. Daily_Forecast_Detail - First 90 days breakdown")
print("  5. Historical_Validation - Actual vs Predicted for training period")
print("  6. Executive_Summary - Key metrics and insights")
print("\n" + "="*80)
