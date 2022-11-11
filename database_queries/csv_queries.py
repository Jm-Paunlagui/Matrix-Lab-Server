import os
import pickle

import pandas as pd

from flask import jsonify, Response
from keras.utils import pad_sequences
from werkzeug.datastructures import FileStorage

from config.configurations import db, app
from models.csv_model import CsvModel
from modules.module import AllowedFile, PayloadSignature, TextPreprocessing
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
    columns_to_not_drop = ["evaluatee", "department", "course_code", "sentence"]

    # @desc: Get the columns that are not required for the evaluation with a seperator of comma
    columns_to_drop = [column for column in reformatted_csv if column not in columns_to_not_drop]

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

    school_year = school_year.replace("S.Y.", "SY").replace(" ", "")
    school_semester = school_semester.replace(" ", "_")
    csv_question = csv_question.title()
    csv_question = csv_question.replace("?", "")
    csv_question = csv_question.replace(" ", "_")

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

    # @desc: Save the csv file to the folder
    csv_to_pred.to_csv(
        app.config["CSV_ANALYZED_FOLDER"] + "/" +
        "ANALYZED-" + csv_question + "_" + school_year
        + "_" + school_semester + ".csv", index=False)

    # @desc: Delete the reformatted csv file from the reformatted folder
    os.remove(os.path.join(app.config["CSV_REFORMATTED_FOLDER"], file_name))

    # @desc: Save the csv file details to the database (csv_name, csv_question, csv_file_path, school_year)
    csv_file = CsvModel(csv_name=file_name, csv_question=csv_question,
                        csv_file_path=app.config["CSV_ANALYZED_FOLDER"] + "/" + "ANALYZED-" + csv_question + "_" +
                        school_year + "_" + school_semester + ".csv", school_year=school_year,
                        school_semester=school_semester)
    db.session.add(csv_file)
    db.session.commit()

    return jsonify({"status": "success",
                    "message": "CSV file evaluated successfully",
                    "csv_file": "ANALYZED-" + csv_question + "_" + school_year + ".csv"}), 200


def count_overall_positive_negative():
    """
    Count the overall positive and negative sentiments.

    :return: The overall positive and negative sentiments
    """
    positive = 0
    negative = 0

    csv_files = CsvModel.query.all()

    for csv_file in csv_files:
        csv_file = pd.read_csv(csv_file.csv_file_path)

        for index, row in csv_file.iterrows():
            if row["sentiment"] >= 50:
                positive += 1
            else:
                negative += 1

    # @desc: Return 0 if there are no positive and negative sentiments
    positive_percentage = round((positive / (positive + negative)) * 100, 2) if positive > 0 else "0"
    negative_percentage = round((negative / (positive + negative)) * 100, 2) if negative > 0 else "0"

    # desc: Starting year and ending year of the csv files
    starting_year = csv_files[0].school_year.split("-")[0] if len(csv_files) > 0 else "----"
    ending_year = csv_files[-1].school_year.split("-")[1] if len(csv_files) > 0 else "----"
    # desc: remove the SY from the school year
    starting_year = starting_year.replace("SY", "") if len(csv_files) > 0 else "----"
    ending_year = ending_year.replace("SY", "") if len(csv_files) > 0 else "-----"

    return positive, negative, positive_percentage, negative_percentage, starting_year, ending_year


nltk.download('stopwords')
# Custom stop words
new_stopwords = [
    "mo", "wla..", "ako", "sa", "akin", "ko", "aking", "sarili", "kami", "atin", "ang", "aming", "lang",
    "amin", "ating", "ka", "iyong", "iyo", "inyong", "siya", "kanya", "mismo", "ito", "nito", "kanyang", "sila",
    "nila",
    "kanila", "kanilang", "kung", "ano", "alin", "sino", "kanino", "na", "mga", "iyon", "am", "ay", "maging",
    "naging",
    "mayroon", "may", "nagkaroon", "pagkakaroon", "gumawa", "ginagawa", "ginawa", "paggawa", "ibig", "dapat",
    "maaari",
    "marapat", "kong", "ikaw", "ta-yo", "namin", "gusto", "nais", "niyang", "nilang", "niya", "huwag", "ginawang",
    "gagawin", "maaaring", "sabihin", "narito", "kapag", "ni", "nasaan", "bakit", "paano", "kailangan", "walang",
    "katiyakan", "isang", "at", "pero", "o", "dahil", "bilang", "hanggang", "habang", "ng", "pamamagitan", "para",
    "tungkol", "laban", "pagitan", "panahon", "bago", "pagkatapos", "itaas", "ibaba", "mula", "pataas", "pababa",
    "palabas", "ibabaw", "ilalim", "muli", "pa", "minsan", "dito", "doon", "saan", "lahat", "anumang", "kapwa",
    "bawat",
    "ilan", "karamihan", "iba", "tulad", "lamang", "pareho", "kaya", "kaysa", "masyado", "napaka", "isa", "bababa",
    "kulang", "marami", "ngayon", "kailanman", "sabi", "nabanggit", "din", "kumuha", "pumunta", "pumupunta",
    "ilagay",
    "makita", "nakita", "katulad", "likod", "kahit", "paraan", "noon", "gayunman", "dalawa", "tatlo", "apat",
    "lima",
    "una", "pangalawa", "yung", "po"
]
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.remove("no")
stpwrd.remove("t")
stpwrd.extend(new_stopwords)


def remove_stopwords(text):
    """
    Remove the stop words from the text.

    :param text: The text to remove the stop words
    :return: The text without the stop words
    """
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stpwrd]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def get_all_the_details_from_csv():
    """
    Get all the details from the csv file. This is used for the admin dashboard.
    """

    # @desc: Read all the csv file in the database
    csv_files = CsvModel.query.all()

    # @desc: Read all the csv file in the database by accessing the csv_file_path column and get the evaluatee column
    # and return a list of evaluatee
    evaluatees = [pd.read_csv(csv_files.csv_file_path)[
                      "evaluatee"].to_list() for csv_files in csv_files]

    # @desc: Flatten the list of evaluatee
    evaluatees = [
        evaluatee for evaluatees in evaluatees for evaluatee in evaluatees]

    # @desc: Remove the duplicates in the list of evaluatee
    evaluatees = list(set(evaluatees))

    # @desc: Read all the csv file in the database by accessing the csv_file_path column and get the department column
    # and return a list of department
    departments = [pd.read_csv(csv_files.csv_file_path)[
                       "department"].to_list() for csv_files in csv_files]

    # @desc: Flatten the list of department
    departments = [
        department for departments in departments for department in departments]

    # @desc: Remove the duplicates in the list of department
    departments = list(set(departments))

    # @desc: Read all the csv file in the database by accessing the csv_file_path column and get the course_code column
    # and return a list of course_code
    course_codes = [pd.read_csv(csv_files.csv_file_path)[
                        "course_code"].to_list() for csv_files in csv_files]

    # @desc: Flatten the list of course_code
    course_codes = [
        course_code for course_codes in course_codes for course_code in course_codes]

    # @desc: Remove the duplicates in the list of course_code
    course_codes = list(set(course_codes))

    # desc: Titles the list of evaluatee, department and course_code
    titles = ["No. of Professors", "No. of Departments",
              "No. of Courses", "No. of CSV Files"]

    # @desc: Get total number of sentences in each csv file and return a list of total number of sentences and also the
    # School Year and School Semester
    # structure of the list: [[total number of sentences, school year, school semester],
    # [total number of sentences, school year, school semester]]
    total_number_of_sentences = [
        [len(pd.read_csv(csv_files.csv_file_path)), csv_files.school_year, csv_files.school_semester]
        for csv_files in csv_files]

    # desc: Overall positive and negative sentiments
    positive, negative, positive_percentage, negative_percentage, starting_year, ending_year = \
        count_overall_positive_negative()

    # @desc: Return the list of evaluatee
    return jsonify({"status": "success",
                    "details": [
                        {"title": titles[0], "value": len(
                            evaluatees), "id": 1, "icon": "fas fa-user-tie"},
                        {"title": titles[1], "value": len(
                            departments), "id": 2, "icon": "fas fa-building"},
                        {"title": titles[2], "value": len(
                            course_codes), "id": 3, "icon": "fas fa-book"},
                        {"title": titles[3], "value": len(
                            csv_files), "id": 4, "icon": "fas fa-file-csv"}
                    ],
                    "evaluatees": [
                        {"name": evaluatee, "id": index + 1} for index, evaluatee in enumerate(evaluatees)
                    ],
                    "departments": [
                        {"department": department, "id": index + 1} for index, department in enumerate(departments)
                    ],
                    "course_codes": [
                        {"course_code": course_code, "id": index + 1} for index, course_code in enumerate(course_codes)
                    ],
                    "csv_files": [
                        {"csv_file": csv_file.csv_name, "id": csv_file.csv_id} for csv_file in csv_files
                    ],
                    "total_responses": [
                        {"total_number_of_responses": total_number_of_sentences[0],
                         "school_year": total_number_of_sentences[1],
                         "school_semester": total_number_of_sentences[2],
                         "id": index + 1} for index, total_number_of_sentences
                        in enumerate(total_number_of_sentences)
                    ],
                    "overall_sentiments": [
                        {"id": 1, "title": "Positive",
                         "value": f"{positive:,}",
                         "percentage": positive_percentage,
                         "year": starting_year + " - " + ending_year,
                         "icon": "fas fa-face-smile-beam"},
                        {"id": 2, "title": "Negative",
                         "value": f"{negative:,}",
                         "percentage": negative_percentage,
                         "year": starting_year + " - " + ending_year,
                         "icon": "fas fa-face-frown"}
                    ]
                    }), 200
