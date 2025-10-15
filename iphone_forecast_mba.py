"""
iPhone Sales Forecasting Model for MBA Operations Management Course
Predicts iPhone 16 and iPhone 17 sales for first 4 months after launch
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# [1/7] DATA LOADING
# ==============================================================================
print("=" * 80)
print("iPhone Sales Forecasting - MBA Operations Management Project")
print("=" * 80)

# Load data
data_path = '/workspaces/Operation-Management/Iphone_Sales_Data(1).csv'
df = pd.read_csv(data_path)

# Clean column names (remove unnamed columns)
df = df[['Date', 'Model', 'Estimated_Units_Millions']]
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(['Model', 'Date']).reset_index(drop=True)

print(f"\n[1/7] Data loaded successfully")
print(f"  - Total records: {len(df):,}")
print(f"  - Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"  - iPhone models: {sorted(df['Model'].unique())}")

# ==============================================================================
# [2/7] MONTHLY AGGREGATION
# ==============================================================================
print(f"\n[2/7] Aggregating daily data to monthly...")

# Add year-month column
df['year_month'] = df['Date'].dt.to_period('M')

# Aggregate to monthly
monthly_df = df.groupby(['Model', 'year_month']).agg({
    'Estimated_Units_Millions': 'sum'
}).reset_index()

# Convert period to datetime (first day of month)
monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
monthly_df = monthly_df.sort_values(['Model', 'date']).reset_index(drop=True)

print(f"  - Monthly records: {len(monthly_df)}")
print(f"  - Sample data:")
print(monthly_df.groupby('Model').head(3))

# ==============================================================================
# [3/7] LAUNCH MONTH CALCULATION
# ==============================================================================
print(f"\n[3/7] Calculating launch_month (months since launch)...")

def calculate_launch_month(group):
    """Calculate launch_month: first non-zero month = 1"""
    # Find first month with sales > 0
    first_nonzero_idx = group[group['Estimated_Units_Millions'] > 0].index.min()

    if pd.isna(first_nonzero_idx):
        # No sales found, set all to NaN
        group['launch_month'] = np.nan
    else:
        # Calculate months since launch (1-indexed)
        first_date = group.loc[first_nonzero_idx, 'date']
        group['launch_month'] = ((group['date'].dt.year - first_date.year) * 12 +
                                  (group['date'].dt.month - first_date.month) + 1)
        # Set launch_month to NaN for months before launch
        group.loc[group['date'] < first_date, 'launch_month'] = np.nan

    return group

monthly_df = monthly_df.groupby('Model', group_keys=False).apply(calculate_launch_month)

# Display launch info
print("  - Launch months identified:")
for model in sorted(monthly_df['Model'].unique()):
    model_data = monthly_df[monthly_df['Model'] == model]
    launch_row = model_data[model_data['launch_month'] == 1]
    if not launch_row.empty:
        print(f"    {model}: {launch_row.iloc[0]['date'].strftime('%Y-%m')}")

# ==============================================================================
# [4/7] FEATURE ENGINEERING
# ==============================================================================
print(f"\n[4/7] Engineering features: month, year, lag1, lag2, ma3, launch_month...")

# Basic time features
monthly_df['month'] = monthly_df['date'].dt.month
monthly_df['year'] = monthly_df['date'].dt.year

# Lag and moving average features (by model)
def create_lag_ma_features(group):
    """Create lag and moving average features per model"""
    # Lag features
    group['lag1'] = group['Estimated_Units_Millions'].shift(1)
    group['lag2'] = group['Estimated_Units_Millions'].shift(2)

    # Moving average (3 months)
    group['ma3'] = group['Estimated_Units_Millions'].rolling(window=3, min_periods=1).mean()

    return group

monthly_df = monthly_df.groupby('Model', group_keys=False).apply(create_lag_ma_features)

# Handle missing values: forward fill for lag features at the beginning
monthly_df['lag1'] = monthly_df.groupby('Model')['lag1'].fillna(method='bfill')
monthly_df['lag2'] = monthly_df.groupby('Model')['lag2'].fillna(method='bfill')

# For ma3, use the available values
monthly_df['ma3'] = monthly_df['ma3'].fillna(monthly_df['Estimated_Units_Millions'])

print("  - Features created:")
print(f"    month: {monthly_df['month'].min()} to {monthly_df['month'].max()}")
print(f"    year: {monthly_df['year'].min()} to {monthly_df['year'].max()}")
print(f"    lag1, lag2, ma3: ✓")
print(f"    launch_month: 1 to {monthly_df['launch_month'].max():.0f}")

# ==============================================================================
# [5/7] MODEL TRAINING - TASK A: iPhone 16 Forecast
# ==============================================================================
print(f"\n[5/7] TASK A: Training LightGBM to predict iPhone 16 (using iPhone 12-15)...")

# Prepare training data (iPhone 12-15)
train_models = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
train_data = monthly_df[monthly_df['Model'].isin(train_models)].copy()

# Remove rows with missing launch_month (before launch)
train_data = train_data[train_data['launch_month'].notna()].copy()

# Features and target
feature_cols = ['month', 'year', 'lag1', 'lag2', 'ma3', 'launch_month']
target_col = 'Estimated_Units_Millions'

X_train = train_data[feature_cols]
y_train = train_data[target_col]

# Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

train_set = lgb.Dataset(X_train, y_train)
model_16 = lgb.train(params, train_set, num_boost_round=200)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Features: {feature_cols}")

# Prepare iPhone 16 actual data for validation
iphone16_actual = monthly_df[monthly_df['Model'] == 'iPhone 16'].copy()
iphone16_actual = iphone16_actual[iphone16_actual['launch_month'].notna()].copy()

# Get first 4 months
iphone16_actual_4m = iphone16_actual[iphone16_actual['launch_month'] <= 4].copy()

# Predict iPhone 16
if len(iphone16_actual_4m) > 0:
    X_test_16 = iphone16_actual_4m[feature_cols]
    iphone16_actual_4m['predicted'] = model_16.predict(X_test_16)

    print(f"  - iPhone 16 actual months available: {len(iphone16_actual_4m)}")
    print(f"  - Launch month range: 1-{iphone16_actual_4m['launch_month'].max():.0f}")
else:
    print("  - WARNING: No iPhone 16 actual data found for validation")

# ==============================================================================
# [5/7] MODEL TRAINING - TASK B: iPhone 17 Forecast
# ==============================================================================
print(f"\n[5/7] TASK B: Training LightGBM to predict iPhone 17 (using iPhone 12-16)...")

# Prepare training data (iPhone 12-16)
train_models_17 = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15', 'iPhone 16']
train_data_17 = monthly_df[monthly_df['Model'].isin(train_models_17)].copy()
train_data_17 = train_data_17[train_data_17['launch_month'].notna()].copy()

X_train_17 = train_data_17[feature_cols]
y_train_17 = train_data_17[target_col]

train_set_17 = lgb.Dataset(X_train_17, y_train_17)
model_17 = lgb.train(params, train_set_17, num_boost_round=200)

print(f"  - Training samples: {len(X_train_17)}")

# Prepare iPhone 17 forecast (simulate first 4 months)
# Assume iPhone 17 launch: 2025-09-01
iphone17_launch_date = pd.to_datetime('2025-09-01')
iphone17_forecast_dates = pd.date_range(iphone17_launch_date, periods=4, freq='MS')

# Create forecast dataframe
iphone17_forecast = pd.DataFrame({
    'Model': 'iPhone 17',
    'date': iphone17_forecast_dates,
    'launch_month': range(1, 5),
    'month': [d.month for d in iphone17_forecast_dates],
    'year': [d.year for d in iphone17_forecast_dates]
})

# For lag features, use iPhone 16's average first 4 months
if len(iphone16_actual_4m) > 0:
    avg_sales_16 = iphone16_actual_4m['Estimated_Units_Millions'].mean()
else:
    avg_sales_16 = train_data_17[train_data_17['launch_month'] <= 4]['Estimated_Units_Millions'].mean()

iphone17_forecast['lag1'] = avg_sales_16
iphone17_forecast['lag2'] = avg_sales_16
iphone17_forecast['ma3'] = avg_sales_16

# Predict iPhone 17
X_test_17 = iphone17_forecast[feature_cols]
iphone17_forecast['predicted'] = model_17.predict(X_test_17)

print(f"  - iPhone 17 forecast months: {len(iphone17_forecast)}")
print(f"  - Launch date: {iphone17_launch_date.strftime('%Y-%m')}")

# ==============================================================================
# [6/7] EVALUATION METRICS (MAPE/WAPE)
# ==============================================================================
print(f"\n[6/7] Calculating MAPE and WAPE for iPhone 16...")

def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_wape(actual, predicted):
    """Weighted Absolute Percentage Error"""
    return np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

if len(iphone16_actual_4m) > 0:
    mape_16 = calculate_mape(iphone16_actual_4m['Estimated_Units_Millions'],
                             iphone16_actual_4m['predicted'])
    wape_16 = calculate_wape(iphone16_actual_4m['Estimated_Units_Millions'],
                             iphone16_actual_4m['predicted'])

    print(f"\n  iPhone 16 (First 4 months after launch):")
    print(f"  - MAPE: {mape_16:.2f}%")
    print(f"  - WAPE: {wape_16:.2f}%")
    print(f"\n  Detailed comparison:")

    comparison_16 = iphone16_actual_4m[['date', 'launch_month',
                                        'Estimated_Units_Millions', 'predicted']].copy()
    comparison_16.columns = ['Date', 'Launch_Month', 'Actual', 'Predicted']
    comparison_16['Error_%'] = ((comparison_16['Predicted'] - comparison_16['Actual']) /
                                 comparison_16['Actual'] * 100)
    print(comparison_16.to_string(index=False))
else:
    print("  - Cannot calculate MAPE/WAPE: No iPhone 16 validation data")
    mape_16 = None
    wape_16 = None

print(f"\n  iPhone 17 Forecast (First 4 months after launch):")
print(f"  Note: No actual data available yet (future forecast)")
print(f"\n  Predicted sales:")
forecast_17_display = iphone17_forecast[['date', 'launch_month', 'predicted']].copy()
forecast_17_display.columns = ['Date', 'Launch_Month', 'Predicted_Sales_Millions']
print(forecast_17_display.to_string(index=False))

# ==============================================================================
# [7/7] SAVE RESULTS AND CREATE VISUALIZATIONS
# ==============================================================================
print(f"\n[7/7] Saving forecast results and creating visualizations...")

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

    output_path_16 = '/workspaces/Operation-Management/iphone_forecast/output/forecast_iphone16.csv'
    iphone16_output.to_csv(output_path_16, index=False)
    print(f"  - iPhone 16 forecast saved to: {output_path_16}")

# Save iPhone 17 forecast
iphone17_output = iphone17_forecast[['date', 'launch_month', 'predicted']].copy()
iphone17_output.columns = ['Date', 'Launch_Month', 'Predicted_Sales_Millions']
iphone17_output['Note'] = 'Future forecast - no actual data yet'

output_path_17 = '/workspaces/Operation-Management/iphone_forecast/output/forecast_iphone17.csv'
iphone17_output.to_csv(output_path_17, index=False)
print(f"  - iPhone 17 forecast saved to: {output_path_17}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: iPhone 16 - Actual vs Predicted
if len(iphone16_actual_4m) > 0:
    ax1 = axes[0]
    x_labels = [f"Month {int(m)}" for m in iphone16_actual_4m['launch_month']]
    x_pos = np.arange(len(x_labels))

    ax1.plot(x_pos, iphone16_actual_4m['Estimated_Units_Millions'],
             marker='o', linewidth=2, markersize=8, label='Actual', color='#2E86DE')
    ax1.plot(x_pos, iphone16_actual_4m['predicted'],
             marker='s', linewidth=2, markersize=8, label='Predicted',
             linestyle='--', color='#EE5A6F')

    ax1.set_xlabel('Launch Month', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sales (Millions of Units)', fontsize=11, fontweight='bold')
    ax1.set_title(f'iPhone 16: Actual vs Predicted\nMAPE: {mape_16:.2f}% | WAPE: {wape_16:.2f}%',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
else:
    axes[0].text(0.5, 0.5, 'No iPhone 16\nvalidation data',
                 ha='center', va='center', fontsize=14)
    axes[0].set_title('iPhone 16: No Data Available', fontsize=12, fontweight='bold')

# Plot 2: iPhone 17 - Forecast
ax2 = axes[1]
x_labels_17 = [f"Month {int(m)}" for m in iphone17_forecast['launch_month']]
x_pos_17 = np.arange(len(x_labels_17))

ax2.plot(x_pos_17, iphone17_forecast['predicted'],
         marker='o', linewidth=2, markersize=8, label='Forecast', color='#10AC84')

ax2.set_xlabel('Launch Month', fontsize=11, fontweight='bold')
ax2.set_ylabel('Sales (Millions of Units)', fontsize=11, fontweight='bold')
ax2.set_title('iPhone 17: Sales Forecast\n(Launch: Sept 2025)',
              fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos_17)
ax2.set_xticklabels(x_labels_17)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(bottom=0)

plt.tight_layout()
plot_path = '/workspaces/Operation-Management/iphone_forecast/output/forecast_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  - Visualization saved to: {plot_path}")

print("\n" + "=" * 80)
print("FORECASTING COMPLETE!")
print("=" * 80)
print(f"\nOutputs:")
print(f"  1. iPhone 16 forecast CSV: {output_path_16}")
print(f"  2. iPhone 17 forecast CSV: {output_path_17}")
print(f"  3. Comparison graph: {plot_path}")

if mape_16 is not None:
    print(f"\nModel Performance (iPhone 16):")
    print(f"  - MAPE: {mape_16:.2f}%")
    print(f"  - WAPE: {wape_16:.2f}%")

    if mape_16 < 5:
        print(f"  - Assessment: Excellent accuracy! ✓")
    elif mape_16 < 10:
        print(f"  - Assessment: Good accuracy ✓")
    elif mape_16 < 20:
        print(f"  - Assessment: Acceptable for launch period")
    else:
        print(f"  - Assessment: Consider model refinement")

print("\n" + "=" * 80)
