import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from engineering_transforming import X_test,y_test,X_train,y_train,engineer_features
from initial_linear_training import baseline_mae
import pandas as pd

# --- STEP 5: Better Model (XGBoost) ---
print("\n--- Step 5: XGBoost Model ---")

# 1. Set up the XGBoost regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=300,       # Number of trees
    learning_rate=0.05,     # Learning rate
    max_depth=6,            # How deep each tree can go
    random_state=42,
    n_jobs=-1               # Use all CPU cores to train faster
)

# 2. Train the model on the Jan-Aug data
print("Training XGBoost... (this might take a few seconds)")
xgb_model.fit(X_train, y_train)

# 3. Predict on the Sep-Oct data
xgb_preds = xgb_model.predict(X_test)

# 4. Calculate the new error rate
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"XGBoost MAE: ${xgb_mae:.2f}")
try:
    print(f"Improvement over baseline: ${baseline_mae - xgb_mae:.2f} per load!")
except NameError:
    print(f"Improvement over baseline: ${196.95 - xgb_mae:.2f} per load!")




# --- STEP 6 & 7: Final Retraining & Output Files ---
print("\n--- Step 6 & 7: Retraining and Final Output ---")

# 1. Load the validation and December files
# (Note: adjust the path to "data/..." if your files are inside a data folder)
df = pd.read_csv("data/train-test.csv", parse_dates=["date"])
val_df = pd.read_csv("data/validation.csv", parse_dates=["date"])
dec_df = pd.read_csv("data/december-chart-inputs.csv", parse_dates=["date"])
val_template = pd.read_csv("data/validation-predictions-template.csv")

# 2. Process the FULL training dataset (Jan-Oct) instead of just Jan-Aug
df_full_feat = engineer_features(df)
X_full = df_full_feat.drop(columns=['posted_rate'])
y_full = df_full_feat['posted_rate']

# 3. Process the validation and December data using the same function from Step 4
val_feat = engineer_features(val_df)
dec_feat = engineer_features(dec_df)

# 4. Align columns to guarantee they match the training features exactly 
# (This fills any missing equipment types in the validation sets with False)
val_feat = val_feat.reindex(columns=X_full.columns, fill_value=False)
dec_feat = dec_feat.reindex(columns=X_full.columns, fill_value=False)

# 5. Retrain the XGBoost model on the FULL dataset
print("Retraining model on the full Jan-Oct dataset...")
xgb_model.fit(X_full, y_full)

# 6. Predict and save validation_predictions.csv
print("Predicting 12,000 validation loads...")
val_template['predicted_rate'] = xgb_model.predict(val_feat)
val_template.to_csv("validation_predictions.csv", index=False)
print("Saved: validation_predictions.csv")

# 7. Predict and save december-chart-inputs.csv
print("Predicting December chart...")
dec_df['predicted_rate'] = xgb_model.predict(dec_feat)
# Save it back to its original location so score.py can find it
dec_df.to_csv("december-chart-inputs.csv", index=False) 
print("Saved: december-chart-inputs.csv")

print("\n✅ Step 6 & 7 Complete! You are ready to run score.py")