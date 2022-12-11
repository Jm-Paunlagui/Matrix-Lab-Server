import pandas as pd

df = pd.read_csv('eval_raw.csv')

print(df)

# drop na
df = df.dropna()

print(df)

# rewrite the csv file
df.to_csv('eval_raw.csv', index=False)
