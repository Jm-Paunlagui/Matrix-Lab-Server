import pandas as pd

df = pd.read_csv("eval_raw_right_one.csv")

# @desc: Sort the dataframe by the column "Course Code"
df.sort_values(by=["department"], inplace=True, ascending=True)

# @desc: Write the sorted dataframe to a csv file
df.to_csv("eval_raw_right_one_12.csv", index=False)
