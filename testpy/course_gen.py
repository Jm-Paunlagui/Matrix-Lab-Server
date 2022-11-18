import pandas as pd

# Course code generator for the course_code column
# for Psychology courses
psychology_course_codes = [
    "Psych 1",
    "Psych 2",
    "Psych 101A",
    "Psych 102",
    "Psych 101B",
    "Psych 104A",
    "Psych 103",
    "Psych 104B",
    "Psych 105",
    "Psych 106",
    "Psych 107",
    "Psych 108",
    "Psych 109E",
    "Psych 110",
    "Psych 111",
    "Psych 112E",
    "Psych Elective 1",
    "Psych 113",
    "Psych 115E",
    "Psych Elective 2",
    "Psych 114",
    "Psych 116",
]

# for Teacher Education courses
teacher_education_course_codes = [
    "Teach 1",
    "Teach 2",
    "Teach 101A",
    "Teach 102",
    "Teach 101B",
    "Teach 104A",
    "Teach 103",
    "Teach 104B",
    "Teach 105",
    "Teach 106",
    "Teach 107",
    "Teach 108",
    "Teach 109E",
    "Teach 110",
    "Teach 111",
    "Teach 112E",
    "Teach Elective 1",
    "Teach 113",
    "Teach 115E",
    "Teach Elective 2",
    "Teach 114",
    "Teach 116",
]

# for Business courses
business_course_codes = [
    "Bus 1",
    "Bus 2",
    "Bus 101A",
    "Bus 102",
    "Bus 101B",
    "Bus 104A",
    "Bus 103",
    "Bus 104B",
    "Bus 105",
    "Bus 106",
    "Bus 107",
    "Bus 108",
    "Bus 109E",
    "Bus 110",
    "Bus 111",
    "Bus 112E",
    "Bus Elective 1",
    "Bus 113",
    "Bus 115E",
    "Bus Elective 2",
    "Bus 114",
    "Bus 116",
]

# for Computer Science courses
computer_science_course_codes = [
    "CS 1",
    "CS 2",
    "CS 101A",
    "CS 102",
    "CS 101B",
    "CS 104A",
    "CS 103",
    "CS 104B",
    "CS 105",
    "CS 106",
    "CS 107",
    "CS 108",
    "CS 109E",
    "CS 110",
    "CS 111",
    "CS 112E",
    "CS Elective 1",
    "CS 113",
    "CS 115E",
    "CS Elective 2",
    "CS 114",
    "CS 116",
]

# @desc: Read the csv file
csv_file = pd.read_csv("eval_raw_right_one_12.csv")

# @desc: Sort the dataframe by the column "Course Code"
csv_file.sort_values(by=["department"], inplace=True, ascending=True)

# @desc: Get the course_code column and alter it with the course codes above for each department
course_code = csv_file["course_code"].to_list()

# @desc: Get the department column
department = csv_file["department"].to_list()

total_course_needed_for_das = 0
total_course_needed_for_teach = 0
total_course_needed_for_bus = 0
total_course_needed_for_cs = 0

for i in range(len(course_code)):
    # Get the total number of courses for each department
    if department[i] == "DAS":
        total_course_needed_for_das += 1
    elif department[i] == "DTE":
        total_course_needed_for_teach += 1
    elif department[i] == "DBA":
        total_course_needed_for_bus += 1
    elif department[i] == "DCI":
        total_course_needed_for_cs += 1

print(
    f"Total number of courses needed for DAS: {total_course_needed_for_das}"
)
print(
    f"Total number of courses needed for DTE: {total_course_needed_for_teach}"
)
print(
    f"Total number of courses needed for DBA: {total_course_needed_for_bus}"
)
print(
    f"Total number of courses needed for DCI: {total_course_needed_for_cs}"
)

# @desc: Create a new array of psychology_course_codes based on the total number of courses needed for DAS repeat the course codes if it is less than the total number of courses needed for DAS
psychology_course_codes_needed = (
    psychology_course_codes * total_course_needed_for_das
)[:total_course_needed_for_das]

# @desc: Create a new array of teacher_education_course_codes based on the total number of courses needed for DTE repeat the course codes if it is less than the total number of courses needed for DTE
teacher_education_course_codes_needed = (
    teacher_education_course_codes * total_course_needed_for_teach
)[:total_course_needed_for_teach]

# @desc: Create a new array of business_course_codes based on the total number of courses needed for DBA repeat the course codes if it is less than the total number of courses needed for DBA
business_course_codes_needed = (
    business_course_codes * total_course_needed_for_bus
)[:total_course_needed_for_bus]

# @desc: Create a new array of computer_science_course_codes based on the total number of courses needed for DCI repeat the course codes if it is less than the total number of courses needed for DCI
computer_science_course_codes_needed = (
    computer_science_course_codes * total_course_needed_for_cs
)[:total_course_needed_for_cs]

# @desc Sort the department column from the csv file inplace
department.sort()

# @desc: New array of course codes based on the department
department_needed = (
    psychology_course_codes_needed
    + business_course_codes_needed
    + computer_science_course_codes_needed
    + teacher_education_course_codes_needed
)


# @desc: Create a new column in the csv file called course_code_needed
csv_file["course_code"] = department_needed

# @desc: Save the csv file
csv_file.to_csv("eval_raw_right_one_112.csv", index=False)
