import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/train-test.csv", parse_dates=["date"])

train_mask = df['date'].dt.month <= 8
test_mask = df['date'].dt.month > 8

df_train = df[train_mask]
df_test = df[test_mask]

print(f"Total rows: {len(df)}")
print(f"Train size (Jan-Aug): {len(df_train)}")
print(f"Test size (Sep-Oct): {len(df_test)}")

# --- STEP 3: Baseline Model ---
print("\n--- Step 3: Baseline Model ---")

lr_model = LinearRegression()

# Train using only 'distance'
lr_model.fit(df_train[['distance']], df_train['posted_rate'])

# Predict on the test set
baseline_preds = lr_model.predict(df_test[['distance']])
baseline_mae = mean_absolute_error(df_test['posted_rate'], baseline_preds)

print(f"Baseline MAE (Distance only): ${baseline_mae:.2f}")