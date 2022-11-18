# from database_queries.csv_queries import professor_analysis
# from modules.module import InputTextValidation
#
# csv_question = "What are the strengths of the instructor in teaching the course?"
# file_name = "eval_raw_right_one_112.csv"
# school_year = "S.Y. 2022-2023"
# selected_column_for_sentence = "3"
# school_semester = "1st Semester"
# csv_file_path = "C:\\Users\\paunl\\Jm-Paunlagui\\Pycharm-Projects\\Matrix-Lab-Server\\csv_files\\analyzed_csv_files/Analyzed_What_Are_The_Strengths_Of_The_Instructor_In_Teaching_The_Course_SY2022-2023_3rd_Semester.csv"
#
# school_year = InputTextValidation(school_year).to_query_school_year()
# school_semester = InputTextValidation(
#     school_semester).to_query_school_semester()
# csv_question = InputTextValidation(csv_question).to_query_csv_question()
#
# professor_analysis(file_name, csv_question,
#                    csv_file_path, school_year, school_semester)
import inspect

print(f"1 {inspect.stack()[0][1]}")
print(f"2 {inspect.stack()[0][2]}")
print(f"3 {inspect.stack()[0][3]}")
print(f"4 {inspect.stack()[0][4]}")
print(f"5 {inspect.stack()[0][4][0].strip()}")
