import pandas as pd

# --- STEP 4: Feature Engineering ---
print("\n--- Step 4: Feature Engineering ---")

def engineer_features(data):
    df_feat = data.copy()
    
    # Extract month and day of week from the date
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    
    # Convert categorical text (equipment, quote_signal) into numerical dummy variables
    df_feat = pd.get_dummies(df_feat, columns=['equipment'])
    
    # Drop IDs and raw string columns the model cannot process
    cols_to_drop = ['load_id', 'date', 'pickup', 'delivery']
    df_feat = df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns])
    
    return df_feat


df = pd.read_csv("data/train-test.csv", parse_dates=["date"])

train_mask = df['date'].dt.month <= 8
test_mask = df['date'].dt.month > 8

df_train = df[train_mask]
df_test = df[test_mask]


# Apply to the train and test splits
df_train_feat = engineer_features(df_train)
df_test_feat = engineer_features(df_test)

# Separate X (features) and y (target)
X_train = df_train_feat.drop(columns=['posted_rate'])
y_train = df_train_feat['posted_rate']

X_test = df_test_feat.drop(columns=['posted_rate'])
y_test = df_test_feat['posted_rate']

# Align columns to ensure test set exactly matches train set 
X_test = X_test.reindex(columns=X_train.columns, fill_value=False)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")