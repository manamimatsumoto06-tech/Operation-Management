#!/usr/bin/env python3
"""
iPhone 16 Sales Forecasting Model
Uses historical sales data from iPhone 12-15 to predict iPhone 16 monthly sales
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading iPhone sales data...")
df = pd.read_csv('Iphone_Sales_Data(1).csv')

# Clean up column names (remove extra commas)
df.columns = df.columns.str.strip()

# Display basic info
print(f"Total records: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nModels in dataset:")
print(df['Model'].value_counts())

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate to monthly level
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_data = df.groupby(['YearMonth', 'Model'])['Estimated_Units_Millions'].sum().reset_index()
monthly_data.columns = ['YearMonth', 'Model', 'Monthly_Units_Millions']

# Extract month number from launch (1 = first month after launch, etc.)
model_launch_dates = {
    'iPhone 12': '2020-10',
    'iPhone 13': '2021-09',
    'iPhone 14': '2022-09',
    'iPhone 15': '2023-09'
}

# Create a dataframe with months since launch for each model
forecast_data = []

for model, launch_month in model_launch_dates.items():
    model_df = monthly_data[monthly_data['Model'] == model].copy()
    model_df = model_df.sort_values('YearMonth')

    launch_period = pd.Period(launch_month, freq='M')
    model_df['MonthsSinceLaunch'] = (model_df['YearMonth'] - launch_period).apply(lambda x: x.n)

    forecast_data.append(model_df[['Model', 'MonthsSinceLaunch', 'Monthly_Units_Millions']])

forecast_df = pd.concat(forecast_data, ignore_index=True)

# Filter to keep only the first 12 months for each model
forecast_df = forecast_df[forecast_df['MonthsSinceLaunch'] >= 0]
forecast_df = forecast_df[forecast_df['MonthsSinceLaunch'] < 12]

print("\n" + "="*60)
print("Monthly Sales Summary by Model (First 12 Months)")
print("="*60)
pivot_table = forecast_df.pivot_table(
    index='MonthsSinceLaunch',
    columns='Model',
    values='Monthly_Units_Millions',
    aggfunc='sum'
)
print(pivot_table)

# Calculate average sales pattern across models
avg_pattern = forecast_df.groupby('MonthsSinceLaunch')['Monthly_Units_Millions'].agg(['mean', 'std']).reset_index()
avg_pattern.columns = ['MonthsSinceLaunch', 'Avg_Units', 'Std_Units']

print("\n" + "="*60)
print("Average Monthly Pattern (Across iPhone 12-15)")
print("="*60)
print(avg_pattern)

# Apply growth factor based on recent trends
# Calculate average first-year sales for each model
model_totals = forecast_df.groupby('Model')['Monthly_Units_Millions'].sum()
print("\n" + "="*60)
print("Total First Year Sales by Model")
print("="*60)
print(model_totals)

# Calculate year-over-year growth
growth_rates = []
models = ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
for i in range(len(models) - 1):
    if models[i] in model_totals.index and models[i+1] in model_totals.index:
        growth = (model_totals[models[i+1]] / model_totals[models[i]] - 1) * 100
        growth_rates.append(growth)
        print(f"{models[i]} -> {models[i+1]}: {growth:.2f}%")

avg_growth_rate = np.mean(growth_rates)
print(f"\nAverage YoY Growth Rate: {avg_growth_rate:.2f}%")

# Generate iPhone 16 forecast
# Assume iPhone 16 launches in September 2024
iphone16_launch = pd.Period('2024-09', freq='M')

# Method 1: Based on average pattern with growth adjustment
print("\n" + "="*60)
print("iPhone 16 Monthly Sales Forecast")
print("="*60)

# Use iPhone 15 as base with modest growth
iphone15_data = forecast_df[forecast_df['Model'] == 'iPhone 15'].sort_values('MonthsSinceLaunch')

forecast_results = []

for month in range(12):
    # Get iPhone 15 actual if available
    iphone15_month = iphone15_data[iphone15_data['MonthsSinceLaunch'] == month]

    if len(iphone15_month) > 0:
        # Use iPhone 15 as base and apply growth
        base_value = iphone15_month['Monthly_Units_Millions'].values[0]
        # Apply a conservative growth (e.g., 3% based on recent trends)
        forecast_value = base_value * (1 + 0.03)
    else:
        # Fall back to average pattern
        avg_value = avg_pattern[avg_pattern['MonthsSinceLaunch'] == month]['Avg_Units'].values
        if len(avg_value) > 0:
            forecast_value = avg_value[0] * (1 + 0.03)
        else:
            forecast_value = 20.0  # Default fallback

    target_month = iphone16_launch + month
    forecast_results.append({
        'Year': target_month.year,
        'Month': target_month.month,
        'YearMonth': str(target_month),
        'MonthsSinceLaunch': month + 1,
        'Forecasted_Units_Millions': round(forecast_value, 2)
    })

forecast_output = pd.DataFrame(forecast_results)

print(forecast_output.to_string(index=False))

# Calculate total
total_forecast = forecast_output['Forecasted_Units_Millions'].sum()
print(f"\nTotal iPhone 16 Forecasted Sales (First 12 Months): {total_forecast:.2f} Million Units")

# Export to Excel
output_file = 'iPhone16_Sales_Forecast.xlsx'
print(f"\n{'='*60}")
print(f"Exporting forecast to Excel: {output_file}")
print(f"{'='*60}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet 1: iPhone 16 Forecast
    forecast_output.to_excel(writer, sheet_name='iPhone16_Forecast', index=False)

    # Sheet 2: Historical Comparison
    comparison_df = pivot_table.reset_index()
    comparison_df['MonthsSinceLaunch'] = comparison_df['MonthsSinceLaunch'] + 1
    comparison_df['iPhone 16 (Forecast)'] = forecast_output['Forecasted_Units_Millions'].values
    comparison_df.to_excel(writer, sheet_name='Historical_Comparison', index=False)

    # Sheet 3: Summary Statistics
    summary_data = {
        'Metric': [
            'iPhone 12 Total (12 months)',
            'iPhone 13 Total (12 months)',
            'iPhone 14 Total (12 months)',
            'iPhone 15 Total (12 months)',
            'iPhone 16 Forecast Total (12 months)',
            'Average YoY Growth Rate',
            'Forecast Method'
        ],
        'Value': [
            f"{model_totals.get('iPhone 12', 0):.2f} M",
            f"{model_totals.get('iPhone 13', 0):.2f} M",
            f"{model_totals.get('iPhone 14', 0):.2f} M",
            f"{model_totals.get('iPhone 15', 0):.2f} M",
            f"{total_forecast:.2f} M",
            f"{avg_growth_rate:.2f}%",
            "iPhone 15 baseline + 3% growth"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # Sheet 4: Raw Monthly Data
    monthly_data_sorted = monthly_data.sort_values(['Model', 'YearMonth'])
    monthly_data_sorted['YearMonth'] = monthly_data_sorted['YearMonth'].astype(str)
    monthly_data_sorted.to_excel(writer, sheet_name='Historical_Monthly_Data', index=False)

print(f"✓ Forecast exported successfully to {output_file}")
print("\nExcel file contains 4 sheets:")
print("  1. iPhone16_Forecast - Monthly forecast for iPhone 16")
print("  2. Historical_Comparison - Side-by-side comparison with iPhone 12-15")
print("  3. Summary - Key statistics and methodology")
print("  4. Historical_Monthly_Data - Raw historical data")
