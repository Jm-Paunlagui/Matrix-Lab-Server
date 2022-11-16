import math
import os
import pickle

import pandas as pd

from flask import jsonify, Response
from keras.utils import pad_sequences
from werkzeug.datastructures import FileStorage

from config.configurations import db, app
from models.csv_model import CsvModel, CsvProfessorModel, CsvDepartmentModel
from modules.module import AllowedFile, PayloadSignature, TextPreprocessing, InputTextValidation
from keras.models import load_model
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, TweetTokenizer
from textblob import TextBlob, Word, Blobber
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def check_csv_name_exists(csv_question: str, school_year: str, school_semester: str) -> bool:
    """
    Check if the csv name exists in the database.

    :param csv_question: The csv question
    :param school_year: The school year
    :param school_semester: The school semester
    :return: True if the csv name exists, else False
    """
    csv = CsvModel.query.filter_by(csv_question=csv_question,
                                   school_year=school_year, school_semester=school_semester).first()
    return True if csv else False


def get_starting_ending_year():
    """
    Get the starting and ending year of the csv files.

    :return: The starting and ending year of the csv files
    """
    csv_files = CsvModel.query.all()

    # desc: Starting year and ending year of the csv files
    starting_year = csv_files[0].school_year.split(
        "-")[0] if len(csv_files) > 0 else "----"
    ending_year = csv_files[-1].school_year.split(
        "-")[1] if len(csv_files) > 0 else "----"
    # desc: remove the SY from the school year
    starting_year = starting_year.replace(
        "SY", "") if len(csv_files) > 0 else "----"
    ending_year = ending_year.replace(
        "SY", "") if len(csv_files) > 0 else "----"

    return starting_year, ending_year


def save_csv(csv_name: str, csv_file_path: str, csv_question: str, csv_file: FileStorage) -> tuple[Response, int]:
    """
    Save the csv file details to the database.

    :param csv_name: The csv name
    :param csv_file_path: The csv file path
    :param csv_question: The csv question
    :param csv_file: The csv file
    :return: The status and message
    """
    # @desc: Save the csv file
    csv_file.save(os.path.join(app.config["CSV_FOLDER"], AllowedFile(
        csv_file.filename).secure_filename()))

    # @desc: Check if the csv file follows the required format: sentence, evaluatee, department and course code.
    csv_file_ = pd.read_csv(
        app.config["CSV_FOLDER"] + "/" + AllowedFile(csv_file.filename).secure_filename())
    csv_columns = csv_file_.columns
    if csv_columns[0] != "sentence" or csv_columns[1] \
            != "evaluatee" or csv_columns[2] != "department" or csv_columns[3] != "course_code":
        # @desc: Delete the csv file if it does not follow the required format
        os.remove(os.path.join(app.config["CSV_FOLDER"], AllowedFile(
            csv_file.filename).secure_filename()))
        return jsonify({"status": "error", "message": "Invalid csv file format"}), 400

    # @desc: Save the csv file details to the database
    csv = CsvModel(csv_name=csv_name, csv_file_path=csv_file_path,
                   csv_question=csv_question)
    db.session.add(csv)
    db.session.commit()
    return jsonify({"status": "success", "message": "File uploaded successfully"}), 200


def view_columns_with_pandas(csv_file_to_view: FileStorage) -> tuple[Response, int]:
    """
    View the csv file columns with pandas.

    :param csv_file_to_view: The csv file to view
    :return: The status and message
    """

    csv_file_to_view.save(os.path.join(app.config["CSV_UPLOADED_FOLDER"], AllowedFile(
        csv_file_to_view.filename).secure_filename()))
    csv_file_ = pd.read_csv(
        app.config["CSV_UPLOADED_FOLDER"] + "/" + AllowedFile(csv_file_to_view.filename).secure_filename())
    csv_columns = csv_file_.columns

    # desc: Check if the first 3 headers are evaluatee, department and course code
    if csv_columns[0] != "evaluatee" or csv_columns[1] != "department" or csv_columns[2] != "course_code":
        # @desc: Delete the csv file if it does not follow the required format
        os.remove(os.path.join(app.config["CSV_UPLOADED_FOLDER"], AllowedFile(
            csv_file_to_view.filename).secure_filename()))
        return jsonify({"status": "error", "message": "Invalid header format"}), 400

    csv_columns_to_return = []
    # @desc: Do not return the first 3 headers since they are not questions to be evaluated
    for i in range(3, len(csv_columns)):
        csv_columns_to_return.append({"id": i, "name": csv_columns[i]})

    csv_columns_payload = {
        "iss": "http://127.0.0.1:5000",
        "sub": "Columns of the csv file",
        "csv_file_name": AllowedFile(csv_file_to_view.filename).secure_filename(),
        "csv_columns": csv_columns_to_return
    }

    csv_columns_token = PayloadSignature(
        payload=csv_columns_payload).encode_payload()

    return jsonify({"status": "success",
                    "message": "File columns viewed successfully",
                    "token_columns": csv_columns_token}), 200


def csv_formatter(file_name: str, sentence_index: int, evaluatee_index: int, department_index: int,
                  course_code_index: int):
    """
    Format the csv file.

    :param file_name: The csv file name
    :param sentence_index: The sentence index
    :param evaluatee_index: The evaluatee index
    :param department_index: The department index
    :param course_code_index: The course code index
    :return: The formatted csv file
    """
    # @desc: Read the csv file and return a pandas dataframe object
    csv_file = pd.read_csv(app.config["CSV_UPLOADED_FOLDER"] + "/" + file_name)

    # @desc: Get all the columns of the csv file
    csv_columns = csv_file.columns

    reformatted_csv = csv_file.rename(columns={
        csv_columns[sentence_index]: "sentence",
        csv_columns[evaluatee_index]: "evaluatee",
        csv_columns[department_index]: "department",
        csv_columns[course_code_index]: "course_code"
    })

    # @desc: Drop the rest of the columns from the csv file that are not required for the evaluation
    columns_to_not_drop = ["sentence",
                           "evaluatee", "department", "course_code"]

    # @desc: Get the columns that are not required for the evaluation with a seperator of comma
    columns_to_drop = [
        column for column in reformatted_csv if column not in columns_to_not_drop]

    reformatted_csv.drop(columns_to_drop, axis=1, inplace=True)

    # desc: Pass 1 is to drop na values and null values in the csv file even if there are values in the other columns
    reformatted_csv.dropna(subset=["sentence"], inplace=True)

    # desc: Pass 2 is to remove the text if it's a single character like 'a', 'b', 'c', etc.
    reformatted_csv = reformatted_csv[reformatted_csv["sentence"].str.len(
    ) > 1]

    # desc: Pass 3 is to Clean the sentences in the csv file and return a list of cleaned sentences
    for index, row in reformatted_csv.iterrows():
        reformatted_csv.at[index, "sentence"] = TextPreprocessing(
            row["sentence"]).clean_text()

    # @desc: Save the reformatted csv file to the database
    reformatted_csv.to_csv(
        app.config["CSV_REFORMATTED_FOLDER"] + "/" + file_name, index=False)

    # @desc: Delete the csv file from the uploaded folder
    os.remove(os.path.join(app.config["CSV_UPLOADED_FOLDER"], file_name))


def csv_formatter_to_evaluate(file_name: str, sentence_index: int):
    """
    Format the csv file to evaluate.

    :param file_name: The csv file name
    :param sentence_index: The sentence index
    :return: The formatted csv file
    """
    # @desc: Read the csv file and return a pandas dataframe object
    csv_file = pd.read_csv(app.config["CSV_UPLOADED_FOLDER"] + "/" + file_name)

    # @desc: Get all the columns of the csv file
    csv_columns = csv_file.columns

    reformatted_csv = csv_file.rename(columns={
        csv_columns[sentence_index]: "sentence"
    })

    # @desc: Drop the rest of the columns from the csv file that are not required for the evaluation
    columns_to_not_drop = ["evaluatee",
                           "department", "course_code", "sentence"]

    # @desc: Get the columns that are not required for the evaluation with a seperator of comma
    columns_to_drop = [
        column for column in reformatted_csv if column not in columns_to_not_drop]

    reformatted_csv.drop(columns_to_drop, axis=1, inplace=True)

    # desc: Pass 1 is to drop na values and null values in the csv file even if there are values in the other columns
    reformatted_csv.dropna(subset=["sentence"], inplace=True)

    # desc: Pass 2 is to remove the text if it's a single character like 'a', 'b', 'c', etc.
    reformatted_csv = reformatted_csv[reformatted_csv["sentence"].str.len(
    ) > 1]

    # desc: Pass 3 is to Clean the sentences in the csv file and return a list of cleaned sentences
    for index, row in reformatted_csv.iterrows():
        reformatted_csv.at[index, "sentence"] = TextPreprocessing(
            row["sentence"]).clean_text()

    # @desc: Save the reformatted csv file to the database
    reformatted_csv.to_csv(
        app.config["CSV_REFORMATTED_FOLDER"] + "/" + file_name, index=False)

    # @desc: Delete the csv file from the uploaded folder
    os.remove(os.path.join(app.config["CSV_UPLOADED_FOLDER"], file_name))


def professor_analysis(csv_name: str, csv_question: str, csv_file_path: str, school_year: str, school_semester: str):
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
    path: str = app.config["CSV_PROFESSOR_ANALYSIS_FOLDER"] + "/" + "Analysis_for_Professors_" + csv_question + "_" + \
        school_year + "_" + school_semester + ".csv"
    # @desc: Save the csv file to the professor_analysis_csv_files folder
    df.to_csv(path, index=False)
    # @desc: Save the details of the professor to the database
    professor_csv = CsvProfessorModel(
        csv_name=csv_name, csv_question=csv_question, csv_file_path=path, school_year=school_year,
        school_semester=school_semester)
    db.session.add(professor_csv)
    db.session.commit()


def department_analysis(csv_name: str, csv_question: str, csv_file_path: str, school_year: str, school_semester: str):
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

    csv_file = pd.read_csv(csv_file_path)

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

    department_list = []
    department_overall_sentiment = []
    department_evaluatee = []
    department_number_of_sentiments = []
    department_positive_sentiments_percentage = []
    department_negative_sentiments_percentage = []

    for index, department in enumerate(average_sentiment_each_department):
        department_list.append(department)
        department_overall_sentiment.append(
            average_sentiment_each_department[department])
        department_evaluatee.append(
            int(csv_file[csv_file["department"] ==
                department]["evaluatee"].nunique())
        )
        department_number_of_sentiments.append(
            len(sentiment_each_department[department]))
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
    path: str = app.config["CSV_DEPARTMENT_ANALYSIS_FOLDER"] + "/" + "Analysis_for_Departments_" + csv_question + "_" \
        + school_year + "_" + school_semester + ".csv"
    # @desc: Save the csv file to the department_analysis_csv_files folder
    df.to_csv(path, index=False)

    # @desc: Save the details of the department to the database
    department_csv = CsvDepartmentModel(
        csv_name=csv_name, csv_question=csv_question, csv_file_path=path, school_year=school_year,
        school_semester=school_semester
    )
    db.session.add(department_csv)
    db.session.commit()


def csv_evaluator(file_name: str, sentence_index: int, school_semester: str, school_year: str, csv_question: str):
    """
    Evaluate the csv file.

    :param file_name: The csv file name
    :param sentence_index: The sentence index
    :param school_semester: The school semester
    :param school_year: The school year
    :param csv_question: The csv question
    :return: The evaluated csv file
    """

    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_school_semester()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    # @desc: Check if the csv file has already been evaluated by csv_question and school_year
    if check_csv_name_exists(csv_question, school_year, school_semester):
        return jsonify({"status": "error", "message": "File already evaluated"}), 409

    # @desc: Format the csv file to the required format: sentence, evaluatee, department and course code.
    csv_formatter_to_evaluate(file_name, sentence_index)

    # @desc: Read the reformatted csv file and return a pandas dataframe object
    csv_to_pred = pd.read_csv(
        app.config["CSV_REFORMATTED_FOLDER"] + "/" + file_name)

    # remove the rows that have empty values in the sentence column
    csv_to_pred = csv_to_pred.dropna(subset=["sentence"])

    tokenizer = pickle.load(open(
        app.config["DEEP_LEARNING_MODEL_FOLDER"] + "/tokenizer.pickle", "rb"))

    model = load_model(
        app.config["DEEP_LEARNING_MODEL_FOLDER"] + "/model.h5")

    # @desc: Get the sentences from the csv file
    sentences = csv_to_pred["sentence"].to_list()

    # @desc: Lowercase
    sentences = [sentence.lower() for sentence in sentences]

    # @desc: Tokenize the sentences
    tokenized_sentences = tokenizer.texts_to_sequences(sentences)

    # @desc: Pad the tokenized sentences
    padded_sentences = pad_sequences(
        tokenized_sentences, maxlen=300, padding='post')

    # @desc: Predict the sentiment of the sentences
    predictions = model.predict(padded_sentences)

    predictions = [round(round(prediction[0], 4) * 100, 2)
                   for prediction in predictions]

    # @desc: Add the predictions to the csv file
    csv_to_pred["sentiment"] = predictions

    # @desc: Path to the csv file
    path: str = app.config["CSV_ANALYZED_FOLDER"] + "/" + "Analyzed_" + csv_question + "_" + school_year + "_" + \
        school_semester + ".csv"
    # @desc: Save the csv file to the folder
    csv_to_pred.to_csv(path, index=False)

    # @desc: Delete the reformatted csv file from the reformatted folder
    os.remove(os.path.join(app.config["CSV_REFORMATTED_FOLDER"], file_name))

    # @desc: Save the csv file details to the database (csv_name, csv_question, csv_file_path, school_year)
    csv_file = CsvModel(csv_name=file_name, csv_question=csv_question, csv_file_path=path,
                        school_year=school_year, school_semester=school_semester)
    db.session.add(csv_file)
    db.session.commit()

    # @desc: For analysis purposes
    professor_analysis(file_name, csv_question,
                       csv_file.csv_file_path, school_year, school_semester)
    department_analysis(file_name, csv_question,
                        csv_file.csv_file_path, school_year, school_semester)

    return jsonify({"status": "success",
                    "message": "CSV file evaluated successfully",
                    "csv_file": "Analyzed_" + csv_question + "_" + school_year + ".csv"}), 200


def read_overall_data_department_analysis_csv_files():
    """
    Count the overall data of the department analysis csv files. This is for the analysis purposes.
    """

    # @desc: Get the csv files that are department analysis csv files
    csv_files = CsvDepartmentModel.query.all()

    sentiment_each_department, department_number_of_sentiments, department_positive_sentiments_percentage, \
        department_negative_sentiments_percentage, department_share, department_evaluatee = {
        }, {}, {}, {}, {}, {}
    for csv_file in csv_files:
        # @desc: Read the csv file
        csv_file = pd.read_csv(csv_file.csv_file_path)

        for index, row in csv_file.iterrows():
            # desc: Sum up the department_overall_sentiment column and divide it by the total number csv files
            if row["department_list"] in sentiment_each_department:
                sentiment_each_department[row["department_list"]
                                          ] += row["department_overall_sentiment"]
            else:
                sentiment_each_department[row["department_list"]
                                          ] = row["department_overall_sentiment"]

            if row["department_list"] in department_number_of_sentiments:
                department_number_of_sentiments[row["department_list"]
                                                ] += row["department_number_of_sentiments"]
            else:
                department_number_of_sentiments[row["department_list"]
                                                ] = row["department_number_of_sentiments"]

            if row["department_list"] in department_positive_sentiments_percentage:
                department_positive_sentiments_percentage[row["department_list"]] += \
                    row["department_positive_sentiments_percentage"]
            else:
                department_positive_sentiments_percentage[row["department_list"]] = \
                    row["department_positive_sentiments_percentage"]

            if row["department_list"] in department_negative_sentiments_percentage:
                department_negative_sentiments_percentage[row["department_list"]] += \
                    row["department_negative_sentiments_percentage"]
            else:
                department_negative_sentiments_percentage[row["department_list"]] = \
                    row["department_negative_sentiments_percentage"]

            if row["department_list"] in department_share:
                department_share[row["department_list"]
                                 ] += row["department_share"]
            else:
                department_share[row["department_list"]
                                 ] = row["department_share"]

            if row["department_list"] in department_evaluatee:
                department_evaluatee[row["department_list"]
                                     ] += row["department_evaluatee"]
            else:
                department_evaluatee[row["department_list"]
                                     ] = row["department_evaluatee"]

    # @desc: Once sentiment_each_department is summed up, divide it by the total number of csv files to get the average
    # and round it to 2 decimal places
    for key, value in sentiment_each_department.items():
        sentiment_each_department[key] = round(value / len(csv_files), 2)

    for key, value in department_positive_sentiments_percentage.items():
        department_positive_sentiments_percentage[key] = round(
            value / len(csv_files), 2)

    for key, value in department_negative_sentiments_percentage.items():
        department_negative_sentiments_percentage[key] = round(
            value / len(csv_files), 2)

    for key, value in department_share.items():
        department_share[key] = round(value / len(csv_files), 2)

    for key, value in department_evaluatee.items():
        # @desc: Get the number of evaluatee per department and divide it by the total number of csv files and
        # return only the whole number (no decimal places)
        department_evaluatee[key] = int(value / len(csv_files))

    # @desc: Sort in descending order
    sentiment_each_department = dict(sorted(
        sentiment_each_department.items(), key=lambda item: item[1], reverse=True))

    # desc: Starting year and ending year of the csv files
    starting_year, ending_year = get_starting_ending_year()

    return jsonify({
        "status": "success",
        "year": f"{starting_year} - {ending_year}",
        "top_department": [
            {
                "id": index,
                "department": department,
                "overall_sentiment": sentiment_each_department[department],
                "number_of_sentiments": department_number_of_sentiments[department],
                "positive_sentiments_percentage": department_positive_sentiments_percentage[department],
                "negative_sentiments_percentage": department_negative_sentiments_percentage[department],
                "share": department_share[department],
                "evaluatee": department_evaluatee[department]
            } for index, department in enumerate(sentiment_each_department)
        ]}), 200


def read_overall_data_professor_analysis_csv_files():
    """
    Count the overall data of the professor analysis csv files. This is for the analysis purposes.
    """

    # @desc: Get the csv files that are professor analysis csv files
    csv_files = CsvProfessorModel.query.all()

    evaluatee_overall_sentiment, evaluatee_number_of_sentiments, evaluatee_positive_sentiments_percentage, \
        evaluatee_negative_sentiments_percentage, evaluatee_share, evaluatee_department = {
        }, {}, {}, {}, {}, {}
    for csv_file in csv_files:
        # @desc: Read the csv file
        csv_file = pd.read_csv(csv_file.csv_file_path)

        for index, row in csv_file.iterrows():
            # desc: Sum up the professor_overall_sentiment column and divide it by the total number csv files
            if row["evaluatee_list"] in evaluatee_overall_sentiment:
                evaluatee_overall_sentiment[row["evaluatee_list"]
                                            ] += row["evaluatee_overall_sentiment"]
            else:
                evaluatee_overall_sentiment[row["evaluatee_list"]
                                            ] = row["evaluatee_overall_sentiment"]

            if row["evaluatee_list"] in evaluatee_number_of_sentiments:
                evaluatee_number_of_sentiments[row["evaluatee_list"]
                                               ] += row["evaluatee_number_of_sentiments"]
            else:
                evaluatee_number_of_sentiments[row["evaluatee_list"]
                                               ] = row["evaluatee_number_of_sentiments"]

            if row["evaluatee_list"] in evaluatee_positive_sentiments_percentage:
                evaluatee_positive_sentiments_percentage[row["evaluatee_list"]] += \
                    row["evaluatee_positive_sentiments_percentage"]
            else:
                evaluatee_positive_sentiments_percentage[row["evaluatee_list"]] = \
                    row["evaluatee_positive_sentiments_percentage"]

            if row["evaluatee_list"] in evaluatee_negative_sentiments_percentage:
                evaluatee_negative_sentiments_percentage[row["evaluatee_list"]] += \
                    row["evaluatee_negative_sentiments_percentage"]
            else:
                evaluatee_negative_sentiments_percentage[row["evaluatee_list"]] = \
                    row["evaluatee_negative_sentiments_percentage"]

            if row["evaluatee_list"] in evaluatee_share:
                evaluatee_share[row["evaluatee_list"]
                                ] += row["evaluatee_share"]
            else:
                evaluatee_share[row["evaluatee_list"]] = row["evaluatee_share"]

            if row["evaluatee_list"] in evaluatee_department:
                evaluatee_department[row["evaluatee_list"]
                                     ] = row["evaluatee_department"]
            else:
                evaluatee_department[row["evaluatee_list"]
                                     ] = row["evaluatee_department"]

    # @desc: Once evaluatee_overall_sentiment is summed up, divide it by the total number of csv files to get
    # the average and round it to 2 decimal places
    for key, value in evaluatee_overall_sentiment.items():
        evaluatee_overall_sentiment[key] = round(value / len(csv_files), 2)

    for key, value in evaluatee_positive_sentiments_percentage.items():
        evaluatee_positive_sentiments_percentage[key] = round(
            value / len(csv_files), 2)

    for key, value in evaluatee_negative_sentiments_percentage.items():
        evaluatee_negative_sentiments_percentage[key] = round(
            value / len(csv_files), 2)

    for key, value in evaluatee_share.items():
        evaluatee_share[key] = round(value / len(csv_files), 2)

    # @desc: Sort in descending order
    evaluatee_overall_sentiment = dict(
        sorted(evaluatee_overall_sentiment.items(), key=lambda item: item[1], reverse=True))

    # desc: Starting year and ending year of the csv files
    starting_year, ending_year = get_starting_ending_year()

    return jsonify({
        "status": "success",
        "year": f"{starting_year} - {ending_year}",
        "top_professors": [
            {
                "id": index,
                "professor": professor,
                "overall_sentiment": evaluatee_overall_sentiment[professor],
                "number_of_sentiments": evaluatee_number_of_sentiments[professor],
                "positive_sentiments_percentage": evaluatee_positive_sentiments_percentage[professor],
                "negative_sentiments_percentage": evaluatee_negative_sentiments_percentage[professor],
                "share": evaluatee_share[professor],
                "evaluatee_department": evaluatee_department[professor]
            } for index, professor in enumerate(evaluatee_overall_sentiment)
        ]}), 200


def read_single_data_department_analysis_csv_files(school_year: str, school_semester: str, csv_question: str):
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_school_semester()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    csv_file = CsvDepartmentModel.query.filter_by(school_year=school_year, school_semester=school_semester,
                                                  csv_question=csv_question).first()

    if csv_file is None:
        return jsonify({"status": "error", "message": "No csv file found."}), 400

    # @desc: Read the csv file
    csv_file = pd.read_csv(csv_file.csv_file_path)

    sentiment_each_department, department_number_of_sentiments, department_positive_sentiments_percentage, \
        department_negative_sentiments_percentage, department_share, department_evaluatee = {
        }, {}, {}, {}, {}, {}

    for index, row in csv_file.iterrows():
        sentiment_each_department[row["department_list"]
                                  ] = row["department_overall_sentiment"]
        department_number_of_sentiments[row["department_list"]
                                        ] = row["department_number_of_sentiments"]
        department_positive_sentiments_percentage[row["department_list"]] = \
            row["department_positive_sentiments_percentage"]
        department_negative_sentiments_percentage[row["department_list"]] = \
            row["department_negative_sentiments_percentage"]
        department_share[row["department_list"]] = row["department_share"]
        department_evaluatee[row["department_list"]
                             ] = row["department_evaluatee"]

    # @desc: Sort in descending order
    sentiment_each_department = dict(
        sorted(sentiment_each_department.items(), key=lambda item: item[1], reverse=True))

    school_year = school_year.replace("SY", "").replace("-", " - ")

    return jsonify({
        "status": "success",
        "year": school_year,
        "top_departments": [
            {
                "id": index,
                "department": department,
                "overall_sentiment": sentiment_each_department[department],
                "number_of_sentiments": department_number_of_sentiments[department],
                "positive_sentiments_percentage": department_positive_sentiments_percentage[department],
                "negative_sentiments_percentage": department_negative_sentiments_percentage[department],
                "share": department_share[department],
                "evaluatee": department_evaluatee[department]
            } for index, department in enumerate(sentiment_each_department)
        ]}), 200


def read_single_data_professor_analysis_csv_files(school_year: str, school_semester: str, csv_question: str):
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_school_semester()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    csv_file = CsvProfessorModel.query.filter_by(school_year=school_year, school_semester=school_semester,
                                                 csv_question=csv_question).first()

    evaluatee_overall_sentiment, evaluatee_number_of_sentiments, evaluatee_positive_sentiments_percentage, \
        evaluatee_negative_sentiments_percentage, evaluatee_share, evaluatee_department = {
        }, {}, {}, {}, {}, {}

    if csv_file is None:
        return jsonify({"status": "error", "message": "No csv file found."}), 400

    # @desc: Read the csv file
    csv_file = pd.read_csv(csv_file.csv_file_path)

    for index, row in csv_file.iterrows():
        evaluatee_overall_sentiment[row["evaluatee_list"]
                                    ] = row["evaluatee_overall_sentiment"]
        evaluatee_number_of_sentiments[row["evaluatee_list"]
                                       ] = row["evaluatee_number_of_sentiments"]
        evaluatee_positive_sentiments_percentage[row["evaluatee_list"]] = row[
            "evaluatee_positive_sentiments_percentage"]
        evaluatee_negative_sentiments_percentage[row["evaluatee_list"]] = row[
            "evaluatee_negative_sentiments_percentage"]
        evaluatee_share[row["evaluatee_list"]] = row["evaluatee_share"]
        evaluatee_department[row["evaluatee_list"]
                             ] = row["evaluatee_department"]

    # @desc: Sort in descending order
    evaluatee_overall_sentiment = dict(
        sorted(evaluatee_overall_sentiment.items(), key=lambda item: item[1], reverse=True))

    school_year = school_year.replace("SY", "").replace("-", " - ")

    return jsonify({
        "status": "success",
        "year": school_year,
        "top_professors": [
            {
                "id": index,
                "professor": professor,
                "overall_sentiment": evaluatee_overall_sentiment[professor],
                "number_of_sentiments": evaluatee_number_of_sentiments[professor],
                "positive_sentiments_percentage": evaluatee_positive_sentiments_percentage[professor],
                "negative_sentiments_percentage": evaluatee_negative_sentiments_percentage[professor],
                "share": evaluatee_share[professor],
                "department": evaluatee_department[professor]
            } for index, professor in enumerate(evaluatee_overall_sentiment)
        ]}), 200
