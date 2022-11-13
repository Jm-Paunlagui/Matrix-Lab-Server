import pandas as pd

from config.configurations import app

file_name = "eval_raw_sample.csv"

# @desc: Read the csv file and return a pandas dataframe object
print(app.config['CSV_FOLDER'])
csv = pd.read_csv(app.config["CSV_FOLDER"] + "/" + file_name)

csv_path = app.config["CSV_FOLDER"] + "/" + file_name

# @desc: Get all the columns of the csv file
csv_columns = csv.columns

# @desc: Get the number of rows of the csv file
csv_rows = csv.shape[0]

print(csv_columns)
print(csv_rows)

# desc: Check if the csv file follows the required format: sentence, evaluatee, department and course code.
if csv_columns[0] != "sentence" or csv_columns[1] \
        != "evaluatee" or csv_columns[2] != "department" or csv_columns[3] != "course_code":  # noqa: E501
    print("Invalid csv file format")
