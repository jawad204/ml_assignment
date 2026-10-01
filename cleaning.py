import pandas as pd


def clean(data, weight_median, market_fallback):
    data = data.copy()

    # weight: fix signs, flag blanks, fill with training median
    data["weight"] = data["weight"].abs()
    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight"] = data["weight"].fillna(weight_median)

    # market_index: flag blanks, fill with the average of the same date
    data["market_index_missing"] = data["market_index"].isna().astype(int)
    daily_mean = data.groupby("date")["market_index"].transform("mean")
    data["market_index"] = data["market_index"].fillna(daily_mean)
    data["market_index"] = data["market_index"].fillna(market_fallback)
    return data


df = pd.read_csv("data/train-test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])

weight_median = df["weight"].abs().median()
market_fallback = df["market_index"].median()

df_clean = clean(df, weight_median, market_fallback)
val_clean = clean(val, weight_median, market_fallback)

print("missing weight:", df_clean["weight"].isna().sum(), val_clean["weight"].isna().sum())
print("missing market_index:", df_clean["market_index"].isna().sum(), val_clean["market_index"].isna().sum())
print("flags:", df_clean["market_index_missing"].sum(), val_clean["market_index_missing"].sum())