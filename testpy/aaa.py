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

    csv_file = pd.read_csv(csv_file_path)

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

    evaluatee_list = []
    evaluatee_overall_sentiment = []
    evaluatee_department = []
    evaluatee_number_of_sentiments = []
    evaluatee_positive_sentiments_percentage = []
    evaluatee_negative_sentiments_percentage = []

    for index, evaluatee in enumerate(average_sentiment_each_professor):
        evaluatee_list.append(evaluatee)
        evaluatee_overall_sentiment.append(
            average_sentiment_each_professor[evaluatee])
        evaluatee_department.append(department_of_each_professor[evaluatee])
        evaluatee_number_of_sentiments.append(
            len(sentiment_each_professor[evaluatee]))
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

    # @desc: Save the csv file to the professor_analysis_csv_files folder
    df.to_csv(app.config["CSV_PROFESSOR_ANALYSIS_FOLDER"] + "/" +
              "Analysis_for_Professors_" + csv_question + "_" + school_year
              + "_" + school_semester + ".csv", index=False)

    # @desc: Save the details of the professor to the database
    professor_csv = CsvProfessorModel(csv_name=csv_name, csv_question=csv_question,
                                      csv_file_path=app.config["CSV_PROFESSOR_ANALYSIS_FOLDER"] + "/"
                                                    + "Analysis_for_Professors_" + csv_question + "_" + school_year +
                                                    "_" + school_semester + ".csv", school_year=school_year,
                                      school_semester=school_semester)
    db.session.add(professor_csv)
    db.session.commit()