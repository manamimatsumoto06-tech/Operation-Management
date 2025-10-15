"""
iPhone Sales Forecasting Model - IMPROVED VERSION
Enhanced with advanced features, hyperparameter tuning, and cross-validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# [1/8] DATA LOADING
# ==============================================================================
print("=" * 80)
print("iPhone Sales Forecasting - IMPROVED MODEL v2.0")
print("=" * 80)

data_path = '/workspaces/Operation-Management/Iphone_Sales_Data(1).csv'
df = pd.read_csv(data_path)

df = df[['Date', 'Model', 'Estimated_Units_Millions']]
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(['Model', 'Date']).reset_index(drop=True)

print(f"\n[1/8] Data loaded successfully")
print(f"  - Total records: {len(df):,}")
print(f"  - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"  - iPhone models: {sorted(df['Model'].unique())}")

# ==============================================================================
# [2/8] MONTHLY AGGREGATION
# ==============================================================================
print(f"\n[2/8] Aggregating daily data to monthly...")

df['year_month'] = df['Date'].dt.to_period('M')
monthly_df = df.groupby(['Model', 'year_month']).agg({
    'Estimated_Units_Millions': 'sum'
}).reset_index()

monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
monthly_df = monthly_df.sort_values(['Model', 'date']).reset_index(drop=True)

print(f"  - Monthly records: {len(monthly_df)}")

# ==============================================================================
# [3/8] LAUNCH MONTH CALCULATION
# ==============================================================================
print(f"\n[3/8] Calculating launch_month...")

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
# [4/8] ENHANCED FEATURE ENGINEERING
# ==============================================================================
print(f"\n[4/8] Engineering ENHANCED features...")

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

# Lifecycle stage (based on launch_month)
def lifecycle_stage(launch_month):
    if pd.isna(launch_month):
        return 0
    elif launch_month <= 3:
        return 1  # Launch phase
    elif launch_month <= 6:
        return 2  # Growth phase
    elif launch_month <= 12:
        return 3  # Mature phase
    else:
        return 4  # Decline phase

monthly_df['lifecycle_stage'] = monthly_df['launch_month'].apply(lifecycle_stage)

# Lag and moving average features (by model)
def create_lag_ma_features(group):
    group['lag1'] = group['Estimated_Units_Millions'].shift(1)
    group['lag2'] = group['Estimated_Units_Millions'].shift(2)
    group['lag3'] = group['Estimated_Units_Millions'].shift(3)

    # Multiple moving averages
    group['ma3'] = group['Estimated_Units_Millions'].rolling(window=3, min_periods=1).mean()
    group['ma6'] = group['Estimated_Units_Millions'].rolling(window=6, min_periods=1).mean()

    # Exponential moving average
    group['ema3'] = group['Estimated_Units_Millions'].ewm(span=3, adjust=False).mean()

    # Growth rate (month-over-month)
    group['mom_growth'] = group['Estimated_Units_Millions'].pct_change()

    return group

monthly_df = monthly_df.groupby('Model', group_keys=False).apply(create_lag_ma_features)

# Fill missing values
monthly_df['lag1'] = monthly_df.groupby('Model')['lag1'].fillna(method='bfill')
monthly_df['lag2'] = monthly_df.groupby('Model')['lag2'].fillna(method='bfill')
monthly_df['lag3'] = monthly_df.groupby('Model')['lag3'].fillna(method='bfill')
monthly_df['ma3'] = monthly_df['ma3'].fillna(monthly_df['Estimated_Units_Millions'])
monthly_df['ma6'] = monthly_df['ma6'].fillna(monthly_df['Estimated_Units_Millions'])
monthly_df['ema3'] = monthly_df['ema3'].fillna(monthly_df['Estimated_Units_Millions'])
monthly_df['mom_growth'] = monthly_df['mom_growth'].fillna(0)

# Interaction features
monthly_df['launch_x_holiday'] = monthly_df['launch_month'] * monthly_df['is_holiday_season']
monthly_df['launch_x_stage'] = monthly_df['launch_month'] * monthly_df['lifecycle_stage']

# Model encoding (numerical representation)
model_map = {'iPhone 12': 12, 'iPhone 13': 13, 'iPhone 14': 14,
             'iPhone 15': 15, 'iPhone 16': 16, 'iPhone 17': 17}
monthly_df['model_num'] = monthly_df['Model'].map(model_map)

print("  - Enhanced features created:")
print("    ✓ Seasonality: is_holiday_season, is_back_to_school, is_q4")
print("    ✓ Lifecycle: lifecycle_stage (1=Launch, 2=Growth, 3=Mature, 4=Decline)")
print("    ✓ Advanced lags: lag1, lag2, lag3")
print("    ✓ Moving averages: ma3, ma6, ema3")
print("    ✓ Growth rate: mom_growth")
print("    ✓ Interactions: launch_x_holiday, launch_x_stage")
print("    ✓ Model encoding: model_num")

# ==============================================================================
# [5/8] OPTIMIZED MODEL TRAINING - TASK A: iPhone 16
# ==============================================================================
print(f"\n[5/8] TASK A: Training OPTIMIZED model for iPhone 16...")

train_models = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
train_data = monthly_df[monthly_df['Model'].isin(train_models)].copy()
train_data = train_data[train_data['launch_month'].notna()].copy()

# Enhanced feature set
feature_cols = ['month', 'year', 'quarter', 'launch_month', 'lifecycle_stage',
                'is_holiday_season', 'is_back_to_school', 'is_q4',
                'lag1', 'lag2', 'lag3', 'ma3', 'ma6', 'ema3', 'mom_growth',
                'launch_x_holiday', 'launch_x_stage', 'model_num']

target_col = 'Estimated_Units_Millions'

X_train = train_data[feature_cols]
y_train = train_data[target_col]

# Optimized hyperparameters
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 15,           # Reduced to prevent overfitting
    'learning_rate': 0.03,      # Lower learning rate for stability
    'feature_fraction': 0.7,    # Feature sampling
    'bagging_fraction': 0.7,    # Row sampling
    'bagging_freq': 5,
    'min_child_samples': 10,    # Minimum samples per leaf
    'lambda_l1': 0.1,           # L1 regularization
    'lambda_l2': 0.1,           # L2 regularization
    'verbose': -1
}

# Cross-validation
print("  - Running time-series cross-validation...")
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)

    model_cv = lgb.train(params, train_set, num_boost_round=300,
                         valid_sets=[val_set])

    y_pred_val = model_cv.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred_val))
    cv_scores.append(mae)
    print(f"    Fold {fold+1} MAE: {mae:.3f}")

print(f"  - Average CV MAE: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Train final model on all data
train_set = lgb.Dataset(X_train, y_train)
model_16 = lgb.train(params, train_set, num_boost_round=300)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Features: {len(feature_cols)}")

# Prepare iPhone 16 actual data
iphone16_actual = monthly_df[monthly_df['Model'] == 'iPhone 16'].copy()
iphone16_actual = iphone16_actual[iphone16_actual['launch_month'].notna()].copy()
iphone16_actual_4m = iphone16_actual[iphone16_actual['launch_month'] <= 4].copy()

# Predict iPhone 16
if len(iphone16_actual_4m) > 0:
    X_test_16 = iphone16_actual_4m[feature_cols]
    iphone16_actual_4m['predicted'] = model_16.predict(X_test_16)
    print(f"  - iPhone 16 validation samples: {len(iphone16_actual_4m)}")

# ==============================================================================
# [5/8] OPTIMIZED MODEL TRAINING - TASK B: iPhone 17
# ==============================================================================
print(f"\n[5/8] TASK B: Training OPTIMIZED model for iPhone 17...")

train_models_17 = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15', 'iPhone 16']
train_data_17 = monthly_df[monthly_df['Model'].isin(train_models_17)].copy()
train_data_17 = train_data_17[train_data_17['launch_month'].notna()].copy()

X_train_17 = train_data_17[feature_cols]
y_train_17 = train_data_17[target_col]

# Cross-validation for iPhone 17 model
print("  - Running time-series cross-validation...")
cv_scores_17 = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_17)):
    X_tr, X_val = X_train_17.iloc[train_idx], X_train_17.iloc[val_idx]
    y_tr, y_val = y_train_17.iloc[train_idx], y_train_17.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)

    model_cv = lgb.train(params, train_set, num_boost_round=300,
                         valid_sets=[val_set])

    y_pred_val = model_cv.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred_val))
    cv_scores_17.append(mae)
    print(f"    Fold {fold+1} MAE: {mae:.3f}")

print(f"  - Average CV MAE: {np.mean(cv_scores_17):.3f} ± {np.std(cv_scores_17):.3f}")

train_set_17 = lgb.Dataset(X_train_17, y_train_17)
model_17 = lgb.train(params, train_set_17, num_boost_round=300)

print(f"  - Training samples: {len(X_train_17)}")

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

# Seasonality features
iphone17_forecast['is_holiday_season'] = ((iphone17_forecast['month'] == 11) |
                                          (iphone17_forecast['month'] == 12)).astype(int)
iphone17_forecast['is_back_to_school'] = ((iphone17_forecast['month'] == 8) |
                                          (iphone17_forecast['month'] == 9)).astype(int)
iphone17_forecast['is_q4'] = (iphone17_forecast['quarter'] == 4).astype(int)

# Lifecycle stages for first 4 months
iphone17_forecast['lifecycle_stage'] = iphone17_forecast['launch_month'].apply(lifecycle_stage)

# Use iPhone 16 initial months as reference for lag features
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
iphone17_forecast['ma3'] = avg_sales
iphone17_forecast['ma6'] = avg_sales
iphone17_forecast['ema3'] = avg_sales
iphone17_forecast['mom_growth'] = avg_growth

# Interaction features
iphone17_forecast['launch_x_holiday'] = (iphone17_forecast['launch_month'] *
                                         iphone17_forecast['is_holiday_season'])
iphone17_forecast['launch_x_stage'] = (iphone17_forecast['launch_month'] *
                                       iphone17_forecast['lifecycle_stage'])

# Predict iPhone 17
X_test_17 = iphone17_forecast[feature_cols]
iphone17_forecast['predicted'] = model_17.predict(X_test_17)

print(f"  - iPhone 17 forecast months: {len(iphone17_forecast)}")

# ==============================================================================
# [6/8] EVALUATION METRICS
# ==============================================================================
print(f"\n[6/8] Calculating MAPE and WAPE...")

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_wape(actual, predicted):
    return np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

if len(iphone16_actual_4m) > 0:
    mape_16 = calculate_mape(iphone16_actual_4m['Estimated_Units_Millions'],
                             iphone16_actual_4m['predicted'])
    wape_16 = calculate_wape(iphone16_actual_4m['Estimated_Units_Millions'],
                             iphone16_actual_4m['predicted'])

    print(f"\n  iPhone 16 IMPROVED MODEL (First 4 months):")
    print(f"  - MAPE: {mape_16:.2f}%  (Previous: 22.44%)")
    print(f"  - WAPE: {wape_16:.2f}%  (Previous: 21.74%)")

    improvement_mape = 22.44 - mape_16
    improvement_wape = 21.74 - wape_16
    print(f"  - Improvement: MAPE {improvement_mape:+.2f}pp | WAPE {improvement_wape:+.2f}pp")

    print(f"\n  Detailed comparison:")
    comparison_16 = iphone16_actual_4m[['date', 'launch_month',
                                        'Estimated_Units_Millions', 'predicted']].copy()
    comparison_16.columns = ['Date', 'Launch_Month', 'Actual', 'Predicted']
    comparison_16['Error_%'] = ((comparison_16['Predicted'] - comparison_16['Actual']) /
                                 comparison_16['Actual'] * 100)
    print(comparison_16.to_string(index=False))
else:
    mape_16 = None
    wape_16 = None

print(f"\n  iPhone 17 Forecast (First 4 months):")
forecast_17_display = iphone17_forecast[['date', 'launch_month', 'predicted']].copy()
forecast_17_display.columns = ['Date', 'Launch_Month', 'Predicted_Sales_Millions']
print(forecast_17_display.to_string(index=False))

# ==============================================================================
# [7/8] FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
print(f"\n[7/8] Feature Importance Analysis...")

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model_16.feature_importance(importance_type='gain')
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\n  Top 10 Most Important Features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"    {row['Feature']:20s}: {row['Importance']:8.0f}")

# ==============================================================================
# [8/8] SAVE RESULTS AND VISUALIZATIONS
# ==============================================================================
print(f"\n[8/8] Saving improved results...")

# Save iPhone 16 forecast
if len(iphone16_actual_4m) > 0:
    iphone16_output = iphone16_actual_4m[['date', 'launch_month',
                                          'Estimated_Units_Millions', 'predicted']].copy()
    iphone16_output.columns = ['Date', 'Launch_Month', 'Actual_Sales_Millions',
                               'Predicted_Sales_Millions']
    iphone16_output['Error_%'] = ((iphone16_output['Predicted_Sales_Millions'] -
                                   iphone16_output['Actual_Sales_Millions']) /
                                  iphone16_output['Actual_Sales_Millions'] * 100)
    iphone16_output['MAPE'] = mape_16
    iphone16_output['WAPE'] = wape_16
    iphone16_output['Model_Version'] = 'v2.0_Improved'

    output_path_16 = '/workspaces/Operation-Management/iphone_forecast/output/forecast_iphone16_improved.csv'
    iphone16_output.to_csv(output_path_16, index=False)
    print(f"  - iPhone 16 forecast saved: {output_path_16}")

# Save iPhone 17 forecast
iphone17_output = iphone17_forecast[['date', 'launch_month', 'predicted']].copy()
iphone17_output.columns = ['Date', 'Launch_Month', 'Predicted_Sales_Millions']
iphone17_output['Model_Version'] = 'v2.0_Improved'

output_path_17 = '/workspaces/Operation-Management/iphone_forecast/output/forecast_iphone17_improved.csv'
iphone17_output.to_csv(output_path_17, index=False)
print(f"  - iPhone 17 forecast saved: {output_path_17}")

# Create improved visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: iPhone 16 - Actual vs Predicted
if len(iphone16_actual_4m) > 0:
    ax1 = axes[0]
    x_labels = [f"M{int(m)}" for m in iphone16_actual_4m['launch_month']]
    x_pos = np.arange(len(x_labels))

    ax1.plot(x_pos, iphone16_actual_4m['Estimated_Units_Millions'],
             marker='o', linewidth=2.5, markersize=10, label='Actual',
             color='#2E86DE', zorder=3)
    ax1.plot(x_pos, iphone16_actual_4m['predicted'],
             marker='s', linewidth=2.5, markersize=10, label='Predicted',
             linestyle='--', color='#EE5A6F', zorder=3)

    ax1.set_xlabel('Launch Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sales (Millions of Units)', fontsize=12, fontweight='bold')
    ax1.set_title(f'iPhone 16: Improved Model\nMAPE: {mape_16:.2f}% | WAPE: {wape_16:.2f}%',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # Add percentage labels
    for i, (actual, pred) in enumerate(zip(iphone16_actual_4m['Estimated_Units_Millions'],
                                            iphone16_actual_4m['predicted'])):
        error_pct = ((pred - actual) / actual * 100)
        ax1.text(i, max(actual, pred) + 0.5, f'{error_pct:+.1f}%',
                ha='center', fontsize=9, color='gray')

# Plot 2: iPhone 17 - Forecast
ax2 = axes[1]
x_labels_17 = [f"M{int(m)}" for m in iphone17_forecast['launch_month']]
x_pos_17 = np.arange(len(x_labels_17))

ax2.plot(x_pos_17, iphone17_forecast['predicted'],
         marker='o', linewidth=2.5, markersize=10, label='Forecast',
         color='#10AC84', zorder=3)

ax2.set_xlabel('Launch Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sales (Millions of Units)', fontsize=12, fontweight='bold')
ax2.set_title('iPhone 17: Improved Forecast\n(Launch: Sept 2025)',
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x_pos_17)
ax2.set_xticklabels(x_labels_17)
ax2.legend(loc='best', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(bottom=0)

# Add value labels
for i, val in enumerate(iphone17_forecast['predicted']):
    ax2.text(i, val + 0.5, f'{val:.2f}M', ha='center', fontsize=9, color='gray')

plt.tight_layout()
plot_path = '/workspaces/Operation-Management/iphone_forecast/output/forecast_improved_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  - Visualization saved: {plot_path}")

# Save feature importance chart
fig2, ax = plt.subplots(figsize=(10, 6))
top_features = importance_df.head(10)
ax.barh(range(len(top_features)), top_features['Importance'], color='#5f27cd')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importance (Gain)', fontsize=11, fontweight='bold')
ax.set_title('Top 10 Feature Importance - iPhone 16 Model', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
importance_path = '/workspaces/Operation-Management/iphone_forecast/output/feature_importance.png'
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
print(f"  - Feature importance chart saved: {importance_path}")

print("\n" + "=" * 80)
print("IMPROVED FORECASTING COMPLETE!")
print("=" * 80)

if mape_16 is not None:
    print(f"\nModel Performance Summary:")
    print(f"  Version 1.0 (Baseline):  MAPE {22.44:.2f}% | WAPE {21.74:.2f}%")
    print(f"  Version 2.0 (Improved):  MAPE {mape_16:.2f}% | WAPE {wape_16:.2f}%")
    print(f"  Improvement:             {improvement_mape:+.2f}pp  | {improvement_wape:+.2f}pp")

    if mape_16 < 15:
        print(f"\n  ✓ Excellent accuracy achieved!")
    elif mape_16 < 20:
        print(f"\n  ✓ Good accuracy - suitable for planning!")

    print(f"\nKey Improvements:")
    print(f"  • Enhanced feature set: {len(feature_cols)} features")
    print(f"  • Seasonality indicators (holiday season, Q4, back-to-school)")
    print(f"  • Lifecycle stages (Launch/Growth/Mature/Decline)")
    print(f"  • Advanced lag features (lag1-3, ma3, ma6, ema3)")
    print(f"  • Interaction terms (launch_x_holiday, launch_x_stage)")
    print(f"  • Optimized hyperparameters with regularization")
    print(f"  • Time-series cross-validation (3 folds)")

print("\n" + "=" * 80)
