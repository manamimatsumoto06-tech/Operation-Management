"""
iPhone 販売予測 - Session 3: 定量的予測（トレンドあり）

授業で扱った線形回帰（Linear Regression）のみを使用したシンプルな予測モデル
- 学習データ: iPhone 12〜15
- テストデータ: iPhone 16（最初4ヶ月）
- モデル: 線形回帰のみ
- 評価指標: MAE, MAPE
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

print("=" * 60)
print("iPhone 販売予測 - 線形回帰モデル（Session 3）")
print("=" * 60)

# ========================================
# 1. データ読み込み
# ========================================
print("\n[1/5] データ読み込み中...")
data = pd.read_csv("Iphone_Sales_Data(1).csv")

# 日付を日付型に変換
data["Date"] = pd.to_datetime(data["Date"])

# 日次データを月次に集計（月の最初の日で集計）
data_monthly = data.groupby(
    ["Model", pd.Grouper(key="Date", freq="MS")]
)["Estimated_Units_Millions"].sum().reset_index()

print(f"✓ データ読み込み完了: {len(data_monthly)} 月次レコード")

# ========================================
# 2. 発売からの経過月数を計算
# ========================================
print("\n[2/5] 特徴量エンジニアリング中...")

# 各モデルの発売月を取得
launch_dates = data_monthly.groupby("Model")["Date"].min().to_dict()

# 発売からの経過月数を計算（1から始まる: 1ヶ月目、2ヶ月目...）
def calculate_months_since_launch(row):
    launch_date = launch_dates[row["Model"]]
    months_diff = (row["Date"].year - launch_date.year) * 12 + \
                  (row["Date"].month - launch_date.month) + 1
    return months_diff

data_monthly["Months_since_launch"] = data_monthly.apply(
    calculate_months_since_launch, axis=1
)

print("✓ 発売からの経過月数を計算")
print(f"  - iPhone 12 発売月: {launch_dates['iPhone 12'].strftime('%Y年%m月')}")
print(f"  - iPhone 13 発売月: {launch_dates['iPhone 13'].strftime('%Y年%m月')}")
print(f"  - iPhone 14 発売月: {launch_dates['iPhone 14'].strftime('%Y年%m月')}")
print(f"  - iPhone 15 発売月: {launch_dates['iPhone 15'].strftime('%Y年%m月')}")
print(f"  - iPhone 16 発売月: {launch_dates['iPhone 16'].strftime('%Y年%m月')}")

# ========================================
# 3. 学習データとテストデータに分割
# ========================================
print("\n[3/5] データ分割中...")

# 学習データ: iPhone 12〜15
train_data = data_monthly[
    data_monthly["Model"].isin(["iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15"])
].copy()

# テストデータ: iPhone 16（最初の4ヶ月のみ）
test_data = data_monthly[
    (data_monthly["Model"] == "iPhone 16") &
    (data_monthly["Months_since_launch"] <= 4)
].copy()

print(f"✓ 学習データ: {len(train_data)} レコード（iPhone 12〜15）")
print(f"✓ テストデータ: {len(test_data)} レコード（iPhone 16、最初4ヶ月）")

# 特徴量（X）とターゲット（y）を準備
# X: 発売からの経過月数（説明変数）
# y: 売上（百万台）（目的変数）
X_train = train_data[["Months_since_launch"]].values
y_train = train_data["Estimated_Units_Millions"].values

X_test = test_data[["Months_since_launch"]].values
y_test = test_data["Estimated_Units_Millions"].values

# ========================================
# 4. 線形回帰モデルの学習
# ========================================
print("\n[4/5] 線形回帰モデルを学習中...")

# 線形回帰モデルを作成
# y = a * x + b の形で、発売からの経過月数から売上を予測
model = LinearRegression()

# モデルを学習データで学習
model.fit(X_train, y_train)

# 学習したモデルのパラメータを表示
print(f"✓ モデル学習完了")
print(f"  - 傾き（係数 a）: {model.coef_[0]:.4f}")
print(f"  - 切片（定数 b）: {model.intercept_:.4f}")
print(f"  - 予測式: 売上 = {model.coef_[0]:.4f} × 経過月数 + {model.intercept_:.4f}")

# ========================================
# 5. iPhone 16 の予測と評価
# ========================================
print("\n[5/5] iPhone 16 を予測中...")

# iPhone 16 の売上を予測
y_pred = model.predict(X_test)

# 誤差指標を計算
mae = mean_absolute_error(y_test, y_pred)

# MAPE（Mean Absolute Percentage Error）を手動計算
# MAPE = (1/n) * Σ|actual - predicted| / actual * 100
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"✓ 予測完了")
print(f"\n【モデル性能】")
print(f"  - MAE（平均絶対誤差）: {mae:.4f} 百万台")
print(f"  - MAPE（平均絶対パーセント誤差）: {mape:.2f} %")

# ========================================
# 6. 結果を表形式で表示
# ========================================
print(f"\n{'='*60}")
print("【iPhone 16 予測結果（最初4ヶ月）】")
print(f"{'='*60}")

# 結果を DataFrame にまとめる
results_df = pd.DataFrame({
    "月": test_data["Date"].dt.strftime("%Y年%m月").values,
    "経過月数": test_data["Months_since_launch"].values,
    "実績（百万台）": y_test,
    "予測（百万台）": y_pred,
    "誤差（百万台）": y_test - y_pred,
    "誤差率（%）": np.abs((y_test - y_pred) / y_test) * 100
})

# 表を見やすく表示
print(results_df.to_string(index=False))

# 合計も表示
print(f"\n{'='*60}")
print(f"合計実績: {y_test.sum():.2f} 百万台")
print(f"合計予測: {y_pred.sum():.2f} 百万台")
print(f"差分: {y_test.sum() - y_pred.sum():.2f} 百万台")
print(f"{'='*60}")

# ========================================
# 7. グラフを作成・保存
# ========================================
print("\n[グラフ作成中...]")

# 日本語フォントの設定（文字化け防止）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# グラフのサイズを設定
fig, ax = plt.subplots(figsize=(10, 6))

# 経過月数を取得
months = test_data["Months_since_launch"].values

# 実績値をプロット（青い実線、丸マーカー）
ax.plot(months, y_test, 'o-', color='blue', linewidth=2,
        markersize=8, label='Actual (Jisseki)', alpha=0.7)

# 予測値をプロット（赤い点線、四角マーカー）
ax.plot(months, y_pred, 's--', color='red', linewidth=2,
        markersize=8, label='Predicted (Yosoku)', alpha=0.7)

# グラフの装飾
ax.set_xlabel('Months since launch (Hatsubai kara no keika tsuki-su)', fontsize=12)
ax.set_ylabel('Sales (Uriage, million units)', fontsize=12)
ax.set_title('iPhone 16 Sales Forecast - Linear Regression\n(Session 3: Quantitative Forecasting with Trend)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# X軸を整数のみに設定
ax.set_xticks(months)

# 性能指標をグラフに追加
textstr = f'MAE: {mae:.4f} million units\nMAPE: {mape:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# グラフを保存
plt.tight_layout()
output_path = "iphone_forecast/output/iphone16_simple_forecast.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ グラフを保存: {output_path}")

# グラフを表示（オプション）
# plt.show()

# ========================================
# 8. CSV 出力
# ========================================
csv_path = "iphone_forecast/output/iphone16_simple_forecast.csv"
results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 結果を保存: {csv_path}")

# ========================================
# 9. iPhone 17 の予測（将来予測）
# ========================================
print("\n" + "="*60)
print("【iPhone 17 予測（2025年9月発売想定）】")
print("="*60)

# iPhone 17の発売後1〜4ヶ月を予測
iphone17_months = np.array([[1], [2], [3], [4]])
iphone17_pred = model.predict(iphone17_months)

# 結果を DataFrame にまとめる
iphone17_results = pd.DataFrame({
    "経過月数": [1, 2, 3, 4],
    "予測月": ["2025年09月", "2025年10月", "2025年11月", "2025年12月"],
    "予測売上（百万台）": iphone17_pred
})

print("\n同じ線形回帰モデルを使用して iPhone 17 を予測:")
print(iphone17_results.to_string(index=False))
print(f"\n合計予測売上: {iphone17_pred.sum():.2f} 百万台")

# ========================================
# 10. iPhone 17 のグラフを作成・保存
# ========================================
print("\n[iPhone 17 グラフ作成中...]")

fig, ax = plt.subplots(figsize=(10, 6))

# 予測値をプロット（赤い点線、丸マーカー）
ax.plot([1, 2, 3, 4], iphone17_pred, 'o--', color='red', linewidth=2,
        markersize=8, label='Predicted (Yosoku)', alpha=0.7)

# グラフの装飾
ax.set_xlabel('Months since launch (Hatsubai kara no keika tsuki-su)', fontsize=12)
ax.set_ylabel('Predicted Sales (Yosoku uriage, million units)', fontsize=12)
ax.set_title('iPhone 17 Forecast - Linear Regression\n(Session 3: Quantitative Forecasting with Trend)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# X軸を整数のみに設定
ax.set_xticks([1, 2, 3, 4])

# モデル情報をグラフに追加
textstr = f'Model: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}\nTotal: {iphone17_pred.sum():.2f} million units'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# グラフを保存
plt.tight_layout()
output_path17 = "iphone_forecast/output/iphone17_simple_forecast.png"
plt.savefig(output_path17, dpi=150, bbox_inches='tight')
print(f"✓ グラフを保存: {output_path17}")

# ========================================
# 11. iPhone 17 の CSV 出力
# ========================================
csv_path17 = "iphone_forecast/output/iphone17_simple_forecast.csv"
iphone17_results.to_csv(csv_path17, index=False, encoding='utf-8-sig')
print(f"✓ 結果を保存: {csv_path17}")

print("\n" + "="*60)
print("完了！すべての処理が正常に終了しました。")
print("  - iPhone 16 実績 vs 予測")
print("  - iPhone 17 将来予測")
print("="*60)
