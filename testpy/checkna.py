import pandas as pd

path = r"C:\Users\paunl\Jm-Paunlagui\Pycharm-Projects\Matrix-Lab-Server\csv_files\reformatted_csv_files\eval_raw.csv"

df = pd.read_csv(path)

print(df.columns)

# Check if there are any empty values in the sentence column
print(df["sentence"].isnull().values.any())

# remove the rows that have empty values in the sentence column or any other column that is required for the evaluation
df = df.dropna(subset=["sentence"])

print(df["sentence"].isnull().values.any())

# write the reformatted csv file to the csv folder
df.to_csv(r"C:\Users\paunl\Jm-Paunlagui\Pycharm-Projects\Matrix-Lab-Server\csv_files\reformatted_csv_files\preprocess_eval_raw.csv", index=False)
