import pandas as pd

from config.configurations import app
from modules.module import TextPreprocessing

file_name = "eval_raw_right_one.csv"

# @desc: Read the csv file and return a pandas dataframe object
csv = pd.read_csv(app.config["CSV_UPLOADED_FOLDER"] + "/" + file_name)

# @desc: Get all the columns of the csv file
csv_columns = csv.columns
print(csv_columns)

# @desc: Reformat the csv file to the required format: sentence, evaluatee, department and course code.

# user will enter the index of the columns that will be renamed to the required format in the csv file
evaluatee_index = 0
department_index = 1
course_code_index = 2
# and the rest of the columns will be dropped from the csv file

reformatted_csv = csv.rename(columns={csv_columns[evaluatee_index]: "evaluatee",
                                      csv_columns[department_index]: "department",
                                      csv_columns[course_code_index]: "course_code"})
# @desc: Drop the COURSE TITLE
reformatted_csv = reformatted_csv.drop(columns=[csv_columns[3]])

print(reformatted_csv.columns)

# @desc: Write the reformatted csv file to the csv folder
reformatted_csv.to_csv(
    app.config["CSV_REFORMATTED_FOLDER"] + "/" + file_name, index=False)


# # @desc: Drop the rest of the columns from the csv file that are not required for the evaluation
# columns_to_not_drop = ["sentence", "evaluatee", "department", "course_code"]
#
# # @desc: Get the columns that are not required for the evaluation with a seperator of comma
# columns_to_drop = [column for column in reformatted_csv if column not in columns_to_not_drop]
#
# # @desc: Drop the columns that are not required for the evaluation
# reformatted_csv = reformatted_csv.drop(columns_to_drop, axis=1)
# print(reformatted_csv.columns)
#
# # @desc: Remove the rows that have empty values in the sentence column
# reformatted_csv = reformatted_csv.dropna(subset=["sentence"])
#
# print(reformatted_csv)
#
# # desc: Pass 1 si to Clean the sentences in the csv file z
# for index, row in reformatted_csv.iterrows():
#     reformatted_csv.at[index, "sentence"] = TextPreprocessing(row["sentence"]).clean_text()
#
# print(reformatted_csv["sentence"])

# # write the reformatted csv file to the csv folder
# reformatted_csv.to_csv(app.config["CSV_REFORMATTED_FOLDER"] + "/" + "preprocess" + "_" + file_name, index=False)
