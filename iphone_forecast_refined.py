"""
iPhone Sales Forecasting Model - REFINED VERSION with Advanced Optimization
Target: MAPE ≤ 15%
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import optuna
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# [1/9] DATA LOADING
# ==============================================================================
print("=" * 80)
print("iPhone Sales Forecasting - REFINED MODEL v3.0")
print("Target: MAPE ≤ 15%")
print("=" * 80)

data_path = '/workspaces/Operation-Management/Iphone_Sales_Data(1).csv'
df = pd.read_csv(data_path)

df = df[['Date', 'Model', 'Estimated_Units_Millions']]
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(['Model', 'Date']).reset_index(drop=True)

print(f"\n[1/9] Data loaded successfully")
print(f"  - Total records: {len(df):,}")
print(f"  - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ==============================================================================
# [2/9] MONTHLY AGGREGATION
# ==============================================================================
print(f"\n[2/9] Aggregating to monthly...")

df['year_month'] = df['Date'].dt.to_period('M')
monthly_df = df.groupby(['Model', 'year_month']).agg({
    'Estimated_Units_Millions': 'sum'
}).reset_index()

monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
monthly_df = monthly_df.sort_values(['Model', 'date']).reset_index(drop=True)

print(f"  - Monthly records: {len(monthly_df)}")

# ==============================================================================
# [3/9] OUTLIER DETECTION AND SMOOTHING (±2σ)
# ==============================================================================
print(f"\n[3/9] Detecting and smoothing outliers (±2σ threshold)...")

def smooth_outliers(group):
    """Detect outliers using ±2σ and replace with rolling mean"""
    sales = group['Estimated_Units_Millions'].values
    mean = sales.mean()
    std = sales.std()

    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std

    outliers = (sales < lower_bound) | (sales > upper_bound)
    n_outliers = outliers.sum()

    if n_outliers > 0:
        # Replace outliers with 3-month rolling average
        sales_series = pd.Series(sales)
        rolling_mean = sales_series.rolling(window=3, center=True, min_periods=1).mean()
        sales[outliers] = rolling_mean[outliers]
        group['Estimated_Units_Millions'] = sales

    return group, n_outliers

total_outliers = 0
smoothed_data = []

for model in monthly_df['Model'].unique():
    model_data = monthly_df[monthly_df['Model'] == model].copy()
    smoothed, n_outliers = smooth_outliers(model_data)
    smoothed_data.append(smoothed)
    total_outliers += n_outliers
    if n_outliers > 0:
        print(f"  - {model}: {n_outliers} outliers smoothed")

monthly_df = pd.concat(smoothed_data, ignore_index=True)
print(f"  - Total outliers smoothed: {total_outliers}")

# ==============================================================================
# [4/9] LAUNCH MONTH CALCULATION
# ==============================================================================
print(f"\n[4/9] Calculating launch_month...")

def calculate_launch_month(group):
    first_nonzero_idx = group[group['Estimated_Units_Millions'] > 0].index.min()
    if pd.isna(first_nonzero_idx):
        group['launch_month'] = np.nan
    else:
        first_date = group.loc[first_nonzero_idx, 'date']
        group['launch_month'] = ((group['date'].dt.year - first_date.year) * 12 +
                                  (group['date'].dt.month - first_date.month) + 1)
        group.loc[group['date'] < first_date, 'launch_month'] = np.nan
    return group

monthly_df = monthly_df.groupby('Model', group_keys=False).apply(calculate_launch_month)

print("  - Launch months identified:")
for model in sorted(monthly_df['Model'].unique()):
    model_data = monthly_df[monthly_df['Model'] == model]
    launch_row = model_data[model_data['launch_month'] == 1]
    if not launch_row.empty:
        print(f"    {model}: {launch_row.iloc[0]['date'].strftime('%Y-%m')}")

# ==============================================================================
# [5/9] ENHANCED FEATURE ENGINEERING WITH LOG TRANSFORMATION
# ==============================================================================
print(f"\n[5/9] Engineering enhanced features with log transformation...")

# Basic time features
monthly_df['month'] = monthly_df['date'].dt.month
monthly_df['year'] = monthly_df['date'].dt.year
monthly_df['quarter'] = monthly_df['date'].dt.quarter

# Seasonality indicators
monthly_df['is_holiday_season'] = ((monthly_df['month'] == 11) |
                                    (monthly_df['month'] == 12)).astype(int)
monthly_df['is_back_to_school'] = ((monthly_df['month'] == 8) |
                                    (monthly_df['month'] == 9)).astype(int)
monthly_df['is_q4'] = (monthly_df['quarter'] == 4).astype(int)

# Lifecycle stage
def lifecycle_stage(launch_month):
    if pd.isna(launch_month):
        return 0
    elif launch_month <= 3:
        return 1  # Launch phase (high volatility)
    elif launch_month <= 6:
        return 2  # Growth phase
    elif launch_month <= 12:
        return 3  # Mature phase
    else:
        return 4  # Decline phase

monthly_df['lifecycle_stage'] = monthly_df['launch_month'].apply(lifecycle_stage)

# Log-transformed features for stability
monthly_df['sales_log'] = np.log1p(monthly_df['Estimated_Units_Millions'])

# Lag and moving average features with log transformation
def create_advanced_features(group):
    # Original scale lags
    group['lag1'] = group['Estimated_Units_Millions'].shift(1)
    group['lag2'] = group['Estimated_Units_Millions'].shift(2)
    group['lag3'] = group['Estimated_Units_Millions'].shift(3)

    # Log-transformed lags (for stability in early launch phase)
    group['lag1_log'] = np.log1p(group['lag1'])
    group['lag2_log'] = np.log1p(group['lag2'])
    group['lag3_log'] = np.log1p(group['lag3'])

    # Moving averages
    group['ma3'] = group['Estimated_Units_Millions'].rolling(window=3, min_periods=1).mean()
    group['ma6'] = group['Estimated_Units_Millions'].rolling(window=6, min_periods=1).mean()

    # Exponential moving average
    group['ema3'] = group['Estimated_Units_Millions'].ewm(span=3, adjust=False).mean()

    # Growth rates
    group['mom_growth'] = group['Estimated_Units_Millions'].pct_change()
    group['mom_growth_ma3'] = group['mom_growth'].rolling(window=3, min_periods=1).mean()

    # Acceleration (change in growth rate)
    group['acceleration'] = group['mom_growth'].diff()

    return group

monthly_df = monthly_df.groupby('Model', group_keys=False).apply(create_advanced_features)

# Fill missing values
for col in ['lag1', 'lag2', 'lag3', 'lag1_log', 'lag2_log', 'lag3_log']:
    monthly_df[col] = monthly_df.groupby('Model')[col].fillna(method='bfill')

monthly_df['ma3'] = monthly_df['ma3'].fillna(monthly_df['Estimated_Units_Millions'])
monthly_df['ma6'] = monthly_df['ma6'].fillna(monthly_df['Estimated_Units_Millions'])
monthly_df['ema3'] = monthly_df['ema3'].fillna(monthly_df['Estimated_Units_Millions'])
monthly_df['mom_growth'] = monthly_df['mom_growth'].fillna(0)
monthly_df['mom_growth_ma3'] = monthly_df['mom_growth_ma3'].fillna(0)
monthly_df['acceleration'] = monthly_df['acceleration'].fillna(0)

# Interaction features
monthly_df['launch_x_holiday'] = monthly_df['launch_month'] * monthly_df['is_holiday_season']
monthly_df['launch_x_stage'] = monthly_df['launch_month'] * monthly_df['lifecycle_stage']
monthly_df['stage_x_holiday'] = monthly_df['lifecycle_stage'] * monthly_df['is_holiday_season']

# Model encoding
model_map = {'iPhone 12': 12, 'iPhone 13': 13, 'iPhone 14': 14,
             'iPhone 15': 15, 'iPhone 16': 16, 'iPhone 17': 17}
monthly_df['model_num'] = monthly_df['Model'].map(model_map)

print("  - Enhanced features with log transformation:")
print("    ✓ Log-transformed lags: lag1_log, lag2_log, lag3_log")
print("    ✓ Original lags: lag1, lag2, lag3")
print("    ✓ Moving averages: ma3, ma6, ema3")
print("    ✓ Growth metrics: mom_growth, mom_growth_ma3, acceleration")
print("    ✓ Interactions: launch_x_holiday, launch_x_stage, stage_x_holiday")

# ==============================================================================
# [6/9] HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ==============================================================================
print(f"\n[6/9] Optimizing hyperparameters with Optuna...")

# Prepare training data
train_models = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
train_data = monthly_df[monthly_df['Model'].isin(train_models)].copy()
train_data = train_data[train_data['launch_month'].notna()].copy()

# Enhanced feature set
feature_cols = ['month', 'year', 'quarter', 'launch_month', 'lifecycle_stage',
                'is_holiday_season', 'is_back_to_school', 'is_q4',
                'lag1', 'lag2', 'lag3', 'lag1_log', 'lag2_log', 'lag3_log',
                'ma3', 'ma6', 'ema3', 'mom_growth', 'mom_growth_ma3', 'acceleration',
                'launch_x_holiday', 'launch_x_stage', 'stage_x_holiday', 'model_num']

target_col = 'Estimated_Units_Millions'

X_train = train_data[feature_cols]
y_train = train_data[target_col]

# Optuna objective function
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
    }

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_set = lgb.Dataset(X_tr, y_tr)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        model = lgb.train(params, train_set, num_boost_round=500,
                         valid_sets=[val_set])

        y_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        scores.append(mae)

    return np.mean(scores)

# Run optimization
print("  - Running Optuna optimization (20 trials)...")
print("  - This may take 1-2 minutes...")

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)

best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'mae'
best_params['verbosity'] = -1

print(f"  - Best CV MAE: {study.best_value:.3f}")
print(f"  - Best parameters:")
for key, value in best_params.items():
    if key not in ['objective', 'metric', 'verbosity']:
        print(f"    {key}: {value}")

# ==============================================================================
# [7/9] TRAIN FINAL MODEL WITH OPTIMIZED PARAMETERS
# ==============================================================================
print(f"\n[7/9] Training final model with optimized parameters...")

train_set = lgb.Dataset(X_train, y_train)
model_16 = lgb.train(best_params, train_set, num_boost_round=500)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Features: {len(feature_cols)}")

# Prepare iPhone 16 actual data
iphone16_actual = monthly_df[monthly_df['Model'] == 'iPhone 16'].copy()
iphone16_actual = iphone16_actual[iphone16_actual['launch_month'].notna()].copy()
iphone16_actual_4m = iphone16_actual[iphone16_actual['launch_month'] <= 4].copy()

# Predict iPhone 16
X_test_16 = iphone16_actual_4m[feature_cols]
iphone16_actual_4m['predicted_refined'] = model_16.predict(X_test_16)

print(f"  - iPhone 16 validation samples: {len(iphone16_actual_4m)}")

# ==============================================================================
# [7/9] TRAIN iPhone 17 MODEL
# ==============================================================================
print(f"\n[7/9] Training iPhone 17 model...")

train_models_17 = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15', 'iPhone 16']
train_data_17 = monthly_df[monthly_df['Model'].isin(train_models_17)].copy()
train_data_17 = train_data_17[train_data_17['launch_month'].notna()].copy()

X_train_17 = train_data_17[feature_cols]
y_train_17 = train_data_17[target_col]

train_set_17 = lgb.Dataset(X_train_17, y_train_17)
model_17 = lgb.train(best_params, train_set_17, num_boost_round=500)

# Prepare iPhone 17 forecast
iphone17_launch_date = pd.to_datetime('2025-09-01')
iphone17_forecast_dates = pd.date_range(iphone17_launch_date, periods=4, freq='MS')

iphone17_forecast = pd.DataFrame({
    'Model': 'iPhone 17',
    'date': iphone17_forecast_dates,
    'launch_month': range(1, 5),
    'month': [d.month for d in iphone17_forecast_dates],
    'year': [d.year for d in iphone17_forecast_dates],
    'quarter': [((d.month - 1) // 3) + 1 for d in iphone17_forecast_dates],
    'model_num': 17
})

# Calculate features for iPhone 17
iphone17_forecast['is_holiday_season'] = ((iphone17_forecast['month'] == 11) |
                                          (iphone17_forecast['month'] == 12)).astype(int)
iphone17_forecast['is_back_to_school'] = ((iphone17_forecast['month'] == 8) |
                                          (iphone17_forecast['month'] == 9)).astype(int)
iphone17_forecast['is_q4'] = (iphone17_forecast['quarter'] == 4).astype(int)
iphone17_forecast['lifecycle_stage'] = iphone17_forecast['launch_month'].apply(lifecycle_stage)

# Use iPhone 16 initial months as reference
if len(iphone16_actual) >= 3:
    ref_data = iphone16_actual[iphone16_actual['launch_month'] <= 3]
    avg_sales = ref_data['Estimated_Units_Millions'].mean()
    avg_growth = ref_data['mom_growth'].mean()
else:
    avg_sales = train_data_17[train_data_17['launch_month'] <= 3]['Estimated_Units_Millions'].mean()
    avg_growth = 0

iphone17_forecast['lag1'] = avg_sales
iphone17_forecast['lag2'] = avg_sales
iphone17_forecast['lag3'] = avg_sales
iphone17_forecast['lag1_log'] = np.log1p(avg_sales)
iphone17_forecast['lag2_log'] = np.log1p(avg_sales)
iphone17_forecast['lag3_log'] = np.log1p(avg_sales)
iphone17_forecast['ma3'] = avg_sales
iphone17_forecast['ma6'] = avg_sales
iphone17_forecast['ema3'] = avg_sales
iphone17_forecast['mom_growth'] = avg_growth
iphone17_forecast['mom_growth_ma3'] = avg_growth
iphone17_forecast['acceleration'] = 0

iphone17_forecast['launch_x_holiday'] = (iphone17_forecast['launch_month'] *
                                         iphone17_forecast['is_holiday_season'])
iphone17_forecast['launch_x_stage'] = (iphone17_forecast['launch_month'] *
                                       iphone17_forecast['lifecycle_stage'])
iphone17_forecast['stage_x_holiday'] = (iphone17_forecast['lifecycle_stage'] *
                                        iphone17_forecast['is_holiday_season'])

X_test_17 = iphone17_forecast[feature_cols]
iphone17_forecast['predicted'] = model_17.predict(X_test_17)

# ==============================================================================
# [8/9] EVALUATION AND COMPARISON
# ==============================================================================
print(f"\n[8/9] Evaluating model performance...")

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_wape(actual, predicted):
    return np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

# Load previous predictions for comparison
# v1.0 baseline and v2.0 improved predictions (recreate from earlier runs)
# For v1.0, we'll use a simple average-based estimate
# For v2.0, we need to re-predict with the v2.0 model

# Recreate v2.0 predictions for comparison
params_v2 = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 15,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1
}

feature_cols_v2 = ['month', 'year', 'quarter', 'launch_month', 'lifecycle_stage',
                   'is_holiday_season', 'is_back_to_school', 'is_q4',
                   'lag1', 'lag2', 'lag3', 'ma3', 'ma6', 'ema3', 'mom_growth',
                   'launch_x_holiday', 'launch_x_stage', 'model_num']

X_train_v2 = train_data[feature_cols_v2]
train_set_v2 = lgb.Dataset(X_train_v2, y_train)
model_16_v2 = lgb.train(params_v2, train_set_v2, num_boost_round=300)

X_test_16_v2 = iphone16_actual_4m[feature_cols_v2]
iphone16_actual_4m['predicted_v2'] = model_16_v2.predict(X_test_16_v2)

# Calculate v1.0 baseline (simple average pattern)
avg_pattern = train_data.groupby('launch_month')['Estimated_Units_Millions'].mean()
iphone16_actual_4m['predicted_v1'] = iphone16_actual_4m['launch_month'].map(avg_pattern)

# Calculate MAPE for all versions
mape_v1 = calculate_mape(iphone16_actual_4m['Estimated_Units_Millions'],
                         iphone16_actual_4m['predicted_v1'])
wape_v1 = calculate_wape(iphone16_actual_4m['Estimated_Units_Millions'],
                         iphone16_actual_4m['predicted_v1'])

mape_v2 = calculate_mape(iphone16_actual_4m['Estimated_Units_Millions'],
                         iphone16_actual_4m['predicted_v2'])
wape_v2 = calculate_wape(iphone16_actual_4m['Estimated_Units_Millions'],
                         iphone16_actual_4m['predicted_v2'])

mape_v3 = calculate_mape(iphone16_actual_4m['Estimated_Units_Millions'],
                         iphone16_actual_4m['predicted_refined'])
wape_v3 = calculate_wape(iphone16_actual_4m['Estimated_Units_Millions'],
                         iphone16_actual_4m['predicted_refined'])

print(f"\n{'='*80}")
print(f"PERFORMANCE COMPARISON - iPhone 16 (First 4 months)")
print(f"{'='*80}")
print(f"\n  Version 1.0 (Baseline):        MAPE {mape_v1:6.2f}% | WAPE {wape_v1:6.2f}%")
print(f"  Version 2.0 (Improved):        MAPE {mape_v2:6.2f}% | WAPE {wape_v2:6.2f}%")
print(f"  Version 3.0 (Refined):         MAPE {mape_v3:6.2f}% | WAPE {wape_v3:6.2f}%")
print(f"\n  Improvement from v1.0 → v3.0:  {mape_v1 - mape_v3:+.2f}pp")
print(f"  Improvement from v2.0 → v3.0:  {mape_v2 - mape_v3:+.2f}pp")

if mape_v3 <= 15:
    print(f"\n  ✓✓✓ TARGET ACHIEVED: MAPE ≤ 15% ✓✓✓")
else:
    print(f"\n  → Target MAPE ≤ 15% (Current: {mape_v3:.2f}%)")

print(f"\n  Detailed comparison:")
comparison = iphone16_actual_4m[['date', 'launch_month', 'Estimated_Units_Millions',
                                  'predicted_v1', 'predicted_v2', 'predicted_refined']].copy()
comparison.columns = ['Date', 'LM', 'Actual', 'Pred_v1.0', 'Pred_v2.0', 'Pred_v3.0']
comparison['Error_v3.0_%'] = ((comparison['Pred_v3.0'] - comparison['Actual']) /
                               comparison['Actual'] * 100)
print(comparison.to_string(index=False))

print(f"\n  iPhone 17 Forecast (Refined v3.0):")
forecast_display = iphone17_forecast[['date', 'launch_month', 'predicted']].copy()
forecast_display.columns = ['Date', 'Launch_Month', 'Predicted_Sales_Millions']
print(forecast_display.to_string(index=False))

# ==============================================================================
# [9/9] SAVE RESULTS AND VISUALIZATIONS
# ==============================================================================
print(f"\n[9/9] Saving refined results...")

# Save iPhone 16 refined forecast
iphone16_output = iphone16_actual_4m[['date', 'launch_month', 'Estimated_Units_Millions',
                                      'predicted_v1', 'predicted_v2', 'predicted_refined']].copy()
iphone16_output.columns = ['Date', 'Launch_Month', 'Actual_Sales_Millions',
                           'Predicted_v1.0', 'Predicted_v2.0', 'Predicted_v3.0_Refined']
iphone16_output['Error_v3.0_%'] = ((iphone16_output['Predicted_v3.0_Refined'] -
                                    iphone16_output['Actual_Sales_Millions']) /
                                   iphone16_output['Actual_Sales_Millions'] * 100)
iphone16_output['MAPE_v1.0'] = mape_v1
iphone16_output['MAPE_v2.0'] = mape_v2
iphone16_output['MAPE_v3.0'] = mape_v3
iphone16_output['WAPE_v3.0'] = wape_v3

output_path_16 = '/workspaces/Operation-Management/iphone_forecast/output/forecast_iphone16_refined.csv'
iphone16_output.to_csv(output_path_16, index=False)
print(f"  - iPhone 16 refined forecast: {output_path_16}")

# Save iPhone 17 forecast
iphone17_output = iphone17_forecast[['date', 'launch_month', 'predicted']].copy()
iphone17_output.columns = ['Date', 'Launch_Month', 'Predicted_Sales_Millions']
iphone17_output['Model_Version'] = 'v3.0_Refined'

output_path_17 = '/workspaces/Operation-Management/iphone_forecast/output/forecast_iphone17_refined.csv'
iphone17_output.to_csv(output_path_17, index=False)
print(f"  - iPhone 17 refined forecast: {output_path_17}")

# Create comprehensive comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: iPhone 16 - All versions comparison
ax1 = axes[0]
x_labels = [f"M{int(m)}" for m in iphone16_actual_4m['launch_month']]
x_pos = np.arange(len(x_labels))

ax1.plot(x_pos, iphone16_actual_4m['Estimated_Units_Millions'],
         marker='o', linewidth=3, markersize=12, label='Actual',
         color='#2E86DE', zorder=4)
ax1.plot(x_pos, iphone16_actual_4m['predicted_v1'],
         marker='x', linewidth=2, markersize=10, label='v1.0 Baseline',
         linestyle=':', color='#95a5a6', alpha=0.7, zorder=1)
ax1.plot(x_pos, iphone16_actual_4m['predicted_v2'],
         marker='s', linewidth=2, markersize=8, label='v2.0 Improved',
         linestyle='--', color='#EE5A6F', alpha=0.8, zorder=2)
ax1.plot(x_pos, iphone16_actual_4m['predicted_refined'],
         marker='D', linewidth=2.5, markersize=9, label='v3.0 Refined',
         linestyle='-', color='#10AC84', zorder=3)

ax1.set_xlabel('Launch Month', fontsize=13, fontweight='bold')
ax1.set_ylabel('Sales (Millions of Units)', fontsize=13, fontweight='bold')
ax1.set_title(f'iPhone 16: Model Comparison\nv1.0: {mape_v1:.2f}%  |  v2.0: {mape_v2:.2f}%  |  v3.0: {mape_v3:.2f}% MAPE',
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.legend(loc='best', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

# Add percentage error labels for v3.0
for i, (actual, pred) in enumerate(zip(iphone16_actual_4m['Estimated_Units_Millions'],
                                        iphone16_actual_4m['predicted_refined'])):
    error_pct = ((pred - actual) / actual * 100)
    color = '#10AC84' if abs(error_pct) < 15 else '#e74c3c'
    ax1.text(i, max(actual, pred) + 0.8, f'{error_pct:+.1f}%',
            ha='center', fontsize=9, color=color, fontweight='bold')

# Plot 2: iPhone 17 - Refined forecast
ax2 = axes[1]
x_labels_17 = [f"M{int(m)}" for m in iphone17_forecast['launch_month']]
x_pos_17 = np.arange(len(x_labels_17))

ax2.plot(x_pos_17, iphone17_forecast['predicted'],
         marker='o', linewidth=2.5, markersize=11, label='Refined Forecast (v3.0)',
         color='#10AC84', zorder=3)

ax2.set_xlabel('Launch Month', fontsize=13, fontweight='bold')
ax2.set_ylabel('Sales (Millions of Units)', fontsize=13, fontweight='bold')
ax2.set_title('iPhone 17: Refined Forecast\n(Launch: Sept 2025)',
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x_pos_17)
ax2.set_xticklabels(x_labels_17)
ax2.legend(loc='best', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(bottom=0)

# Add value labels
for i, val in enumerate(iphone17_forecast['predicted']):
    ax2.text(i, val + 0.5, f'{val:.2f}M', ha='center', fontsize=10,
            color='#10AC84', fontweight='bold')

plt.tight_layout()
plot_path = '/workspaces/Operation-Management/iphone_forecast/output/forecast_refined_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  - Comparison visualization: {plot_path}")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model_16.feature_importance(importance_type='gain')
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print(f"\n  Top 10 Most Important Features (v3.0):")
for idx, row in importance_df.head(10).iterrows():
    print(f"    {row['Feature']:25s}: {row['Importance']:8.0f}")

print("\n" + "=" * 80)
print("REFINED FORECASTING COMPLETE!")
print("=" * 80)
print(f"\nKey Improvements in v3.0:")
print(f"  • Outlier smoothing (±2σ): {total_outliers} outliers corrected")
print(f"  • Log-transformed lag features for launch phase stability")
print(f"  • Enhanced features: {len(feature_cols)} total features")
print(f"  • Hyperparameter optimization via Optuna (20 trials)")
print(f"  • Optimized parameters: num_leaves={best_params['num_leaves']}, "
      f"lr={best_params['learning_rate']:.3f}")
print("\n" + "=" * 80)
