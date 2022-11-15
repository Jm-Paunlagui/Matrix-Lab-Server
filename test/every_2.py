import pandas as pd


def professor_analysis(file_path: str):
    """
    evaluatee_list: The list of the professors without duplicates
    evaluatee_overall_sentiment: The overall sentiment of the professor
    evaluatee_department: The department of the professor
    evaluatee_number_of_sentiments: The number of sentiments of the professor
    evaluatee_positive_sentiments_percentage: The positive sentiments percentage of the professor
    evaluatee_negative_sentiments_percentage: The negative sentiments percentage of the professor
    evaluatee_share: The share of the professor in the total responses of the students
    """

    # @desc: Get the sentiment of each professor
    sentiment_each_professor = {}

    # desc: The department of each professor on were they are teaching
    department_of_each_professor = {}

    # @desc: Get the average sentiment of each professor
    average_sentiment_each_professor = {}

    csv_file = pd.read_csv(file_path)

    for index, row in csv_file.iterrows():
        if row["evaluatee"] not in sentiment_each_professor:
            sentiment_each_professor[row["evaluatee"]] = [row["sentiment"]]
            department_of_each_professor[row["evaluatee"]] = row["department"]
        else:
            sentiment_each_professor[row["evaluatee"]].append(
                row["sentiment"])

    for evaluatee, sentiment in sentiment_each_professor.items():
        average_sentiment_each_professor[evaluatee] = round(
            sum(sentiment) / len(sentiment), 2)

    # @desc: Sort the average sentiment of each professor in descending order
    average_sentiment_each_professor = dict(sorted(average_sentiment_each_professor.items(),
                                                   key=lambda item: item[1], reverse=True))

    print(average_sentiment_each_professor)

    evaluatee_list = []
    evaluatee_overall_sentiment = []
    evaluatee_department = []
    evaluatee_number_of_sentiments = []
    evaluatee_positive_sentiments_percentage = []
    evaluatee_negative_sentiments_percentage = []

    for index, evaluatee in enumerate(average_sentiment_each_professor):
        evaluatee_list.append(evaluatee)
        evaluatee_overall_sentiment.append(average_sentiment_each_professor[evaluatee])
        evaluatee_department.append(department_of_each_professor[evaluatee])
        evaluatee_number_of_sentiments.append(len(sentiment_each_professor[evaluatee]))
        evaluatee_positive_sentiments_percentage.append(
            round((len([sentiment for sentiment in sentiment_each_professor[evaluatee]
                        if sentiment >= 50]) / len(sentiment_each_professor[evaluatee])) * 100, 2))
        evaluatee_negative_sentiments_percentage.append(
            round((len([sentiment for sentiment in sentiment_each_professor[evaluatee]
                        if sentiment < 50]) / len(sentiment_each_professor[evaluatee])) * 100, 2))

    # @desc: Get the share of the professor in the total responses of the students
    evaluatee_share = []
    for index, evaluatee in enumerate(evaluatee_list):
        evaluatee_share.append(
            round((evaluatee_number_of_sentiments[index] / sum(evaluatee_number_of_sentiments)) * 100, 2))

    # @desc: Create a dataframe
    df = pd.DataFrame({
        "evaluatee_list": evaluatee_list,
        "evaluatee_overall_sentiment": evaluatee_overall_sentiment,
        "evaluatee_department": evaluatee_department,
        "evaluatee_number_of_sentiments": evaluatee_number_of_sentiments,
        "evaluatee_positive_sentiments_percentage": evaluatee_positive_sentiments_percentage,
        "evaluatee_negative_sentiments_percentage": evaluatee_negative_sentiments_percentage,
        "evaluatee_share": evaluatee_share
    })

    # @desc: Add the dataframe to the current csv file and save it
    csv_file = pd.concat([csv_file, df], axis=1)
    csv_file.to_csv(file_path, index=False)
def department_analysis(file_path: str):
    """
    department_list: The list of the professors without duplicates
    department_overall_sentiment: The overall sentiment of the professor
    department_evaluatee: The number of evaluatee per department
    department_number_of_sentiments: The number of sentiments of the professor
    department_positive_sentiments_percentage: The positive sentiments percentage of the professor
    department_negative_sentiments_percentage: The negative sentiments percentage of the professor
    department_share: The share of the professor in the total responses of the students
    """

    # @desc: Get the sentiment of each department
    sentiment_each_department = {}

    # @desc: Get the average sentiment of each department
    average_sentiment_each_department = {}

    csv_file = pd.read_csv(file_path)

    for index, row in csv_file.iterrows():
        if row["department"] not in sentiment_each_department:
            sentiment_each_department[row["department"]] = [row["sentiment"]]
        else:
            sentiment_each_department[row["department"]].append(
                row["sentiment"])

    for department, sentiment in sentiment_each_department.items():
        average_sentiment_each_department[department] = round(
            sum(sentiment) / len(sentiment), 2)

    # @desc: Sort the average sentiment of each department in descending order
    average_sentiment_each_department = dict(sorted(average_sentiment_each_department.items(),
                                                    key=lambda item: item[1], reverse=True))

    print(average_sentiment_each_department)

    department_list = []
    department_overall_sentiment = []
    department_evaluatee = []
    department_number_of_sentiments = []
    department_positive_sentiments_percentage = []
    department_negative_sentiments_percentage = []

    for index, department in enumerate(average_sentiment_each_department):
        department_list.append(department)
        department_overall_sentiment.append(average_sentiment_each_department[department])
        # count the number of evaluatee per department and append it to the list of the departments by integer not float
        department_evaluatee.append(int(csv_file[csv_file["department"] == department]["evaluatee"].nunique()))
        department_number_of_sentiments.append(len(sentiment_each_department[department]))
        department_positive_sentiments_percentage.append(
            round((len([sentiment for sentiment in sentiment_each_department[department]
                        if sentiment >= 50]) / len(sentiment_each_department[department])) * 100, 2))
        department_negative_sentiments_percentage.append(
            round((len([sentiment for sentiment in sentiment_each_department[department]
                        if sentiment < 50]) / len(sentiment_each_department[department])) * 100, 2))

    # @desc: Get the share of the professor in the total responses of the students
    department_share = []
    for index, department in enumerate(department_list):
        department_share.append(
            round((department_number_of_sentiments[index] / sum(department_number_of_sentiments)) * 100, 2))

    # @desc: Create a dataframe
    df = pd.DataFrame({
        "department_list": department_list,
        "department_overall_sentiment": department_overall_sentiment,
        "department_evaluatee": department_evaluatee,
        "department_number_of_sentiments": department_number_of_sentiments,
        "department_positive_sentiments_percentage": department_positive_sentiments_percentage,
        "department_negative_sentiments_percentage": department_negative_sentiments_percentage,
        "department_share": department_share
    })

    # @desc: Add the dataframe to the current csv file and save it again or overwrite it
    csv_file = pd.concat([csv_file, df], axis=1)
    csv_file.to_csv(file_path, index=False)
    print(csv_file)


professor_analysis(
    "C:\\Users\\PC-165\\PycharmProjects\\Matrix-Lab-Server\\csv_files\\analyzed_csv_files/ANALYZED-What_Are_The_Strengths_Of_The_Instructor_In_Teaching_The_Course_SY2022-2023_1st_Semester.csv")
