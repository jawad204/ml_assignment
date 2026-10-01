import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("data/train-test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])
dec = pd.read_csv("data/december-chart-inputs.csv")

print(dec.shape)
print(dec.head(3))
print()
print("train dates:", df["date"].min(), "->", df["date"].max())
print("val dates:  ", val["date"].min(), "->", val["date"].max())
print()
print(df.head(3))
print(df[["distance", "weight", "market_index", "quote_signal", "posted_rate"]].describe())