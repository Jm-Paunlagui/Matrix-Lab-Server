import inspect
import os
import pickle
import shutil
import sys
import time
from io import BytesIO

import nltk
import pandas as pd
from flask import jsonify, Response, session, send_file
from nltk import word_tokenize
from sqlalchemy import update, func
from textblob import TextBlob
from werkzeug.datastructures import FileStorage
from keras.utils import pad_sequences

from by_database.models.csv_file import CsvModelDetail, CsvAnalyzedSentiment, CsvCourses
from config import Directories
from extensions import db
from matrix.models.csv_file import CsvErrorModel, CsvModel, CsvProfessorModel, CsvDepartmentModel, CsvCollectionModel, \
    CsvTimeElapsed
from matrix.models.user import User
from matrix.module import AllowedFile, PayloadSignature, TextPreprocessing, InputTextValidation, error_message, \
    Timezone, get_starting_ending_year

nltk.download('punkt')
nltk.download('stopwords')

new_stopwords = [
    "mo", "wla..", "ako", "sa", "akin", "ko", "aking", "sarili", "kami", "atin", "ang", "aming", "lang",
    "amin", "ating", "ka", "iyong", "iyo", "inyong", "siya", "kanya", "mismo", "ito", "nito", "kanyang", "sila", "nila",
    "kanila", "kanilang", "kung", "ano", "alin", "sino", "kanino", "na", "mga", "iyon", "am", "ay", "maging", "naging",
    "mayroon", "may", "nagkaroon", "pagkakaroon", "gumawa", "ginagawa", "ginawa", "paggawa", "ibig", "dapat", "maaari",
    "marapat", "kong", "ikaw", "tayo", "namin", "gusto", "nais", "niyang", "nilang", "niya", "huwag", "ginawang",
    "gagawin", "maaaring", "sabihin", "narito", "kapag", "ni", "nasaan", "bakit", "paano", "kailangan", "walang",
    "katiyakan", "isang", "at", "pero", "o", "dahil", "bilang", "hanggang", "habang", "ng", "pamamagitan", "para",
    "tungkol", "laban", "pagitan", "panahon", "bago", "pagkatapos", "itaas", "ibaba", "mula", "pataas", "pababa",
    "palabas", "ibabaw", "ilalim", "muli", "pa", "minsan", "dito", "doon", "saan", "lahat", "anumang", "kapwa", "bawat",
    "ilan", "karamihan", "iba", "tulad", "lamang", "pareho", "kaya", "kaysa", "masyado", "napaka", "isa", "bababa",
    "kulang", "marami", "ngayon", "kailanman", "sabi", "nabanggit", "din", "kumuha", "pumunta", "pumupunta", "ilagay",
    "makita", "nakita", "katulad", "likod", "kahit", "paraan", "noon", "gayunman", "dalawa", "tatlo", "apat", "lima",
    "una", "pangalawa", "yung", "po"
]
stpwrd = nltk.corpus.stopwords.words('english')
stpwrd.remove("no")
stpwrd.remove("t")
stpwrd.extend(new_stopwords)


def check_csv_name_exists(csv_question: str, school_year: str, school_semester: str) -> bool:
    """
    Check if the csv name exists in the database.

    :param csv_question: The csv question
    :param school_year: The school year
    :param school_semester: The school semester
    :return: True if the csv name exists, else False
    """
    csv = CsvModelDetail.query.filter_by(csv_question=csv_question,
                                         school_year=school_year, school_semester=school_semester).first()
    return bool(csv)


def error_handler(error_occurred: str, name_of: str):
    """
    Log the error to the database.

    :param error_occurred: The error occurred
    :param name_of: The name of the error
    """
    db.session.add(CsvErrorModel(csv_error=error_occurred, name_of=name_of))
    db.session.commit()
    return jsonify({"status": "error", "message": error_occurred}), 500


# def get_starting_ending_year():
#     """
#     Get the starting and ending year of the csv files.
#
#     :return: The starting and ending year of the csv files
#     """
#     csv_files = CsvModelDetail.query.all()
#
#     # desc: Starting year and ending year of the csv files
#     starting_year = csv_files[0].school_year.split(
#         "-")[0] if len(csv_files) > 0 else "----"
#     ending_year = csv_files[-1].school_year.split(
#         "-")[1] if len(csv_files) > 0 else "----"
#     # desc: remove the SY from the school year
#     starting_year = starting_year.replace(
#         "SY", "") if len(csv_files) > 0 else "----"
#     ending_year = ending_year.replace(
#         "SY", "") if len(csv_files) > 0 else "----"
#
#     return starting_year, ending_year


def view_columns_with_pandas(csv_file_to_view: FileStorage) -> tuple[Response, int]:
    """
    View the csv file columns with pandas.

    :param csv_file_to_view: The csv file to view
    :return: The status and message
    """
    csv_file_to_view.save(os.path.join(Directories.CSV_UPLOADED_FOLDER, AllowedFile(
        csv_file_to_view.filename).secure_filename()))
    csv_file_ = pd.read_csv(
        Directories.CSV_UPLOADED_FOLDER + "/" + AllowedFile(csv_file_to_view.filename).secure_filename())
    csv_columns = csv_file_.columns

    # desc: Check if the first 4 headers are evaluatee, email, department and course code
    if csv_columns[0] != "evaluatee" or csv_columns[1] != "email" or csv_columns[2] != \
            "department" or csv_columns[3] != "course_code":
        # @desc: Delete the csv file if it does not follow the required format
        os.remove(os.path.join(Directories.CSV_UPLOADED_FOLDER, AllowedFile(
            csv_file_to_view.filename).secure_filename()))
        return jsonify({"status": "error", "message": "Invalid header format"}), 400

    csv_columns_to_return = []
    # @desc: Do not return the first 4 headers since they are not questions to be evaluated
    for i in range(4, len(csv_columns)):
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
    csv_file = pd.read_csv(Directories.CSV_UPLOADED_FOLDER + "/" + file_name)

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
        Directories.CSV_REFORMATTED_FOLDER + "/" + file_name, index=False)

    # @desc: Delete the csv file from the uploaded folder
    os.remove(os.path.join(Directories.CSV_UPLOADED_FOLDER, file_name))


def csv_formatter_to_evaluate(file_name: str, sentence_index: int):
    """
    Format the csv file to evaluate.

    :param file_name: The csv file name
    :param sentence_index: The sentence index
    :return: The formatted csv file
    """
    # @desc: Read the csv file and return a pandas dataframe object
    csv_file = pd.read_csv(Directories.CSV_UPLOADED_FOLDER + "/" + file_name)

    # @desc: Get all the columns of the csv file
    csv_columns = csv_file.columns

    reformatted_csv = csv_file.rename(columns={
        csv_columns[sentence_index]: "sentence"
    })

    # @desc: Drop the rest of the columns from the csv file that are not required for the evaluation
    columns_to_not_drop = ["evaluatee", "email",
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
        Directories.CSV_REFORMATTED_FOLDER + "/" + file_name, index=False)

    # @desc: Delete the csv file from the uploaded folder
    # os.remove(os.path.join(app.config["CSV_UPLOADED_FOLDER"], file_name))


def done_in_csv_evaluation(file_name: str):
    """
    Delete the uploaded csv file after evaluation.

    :param file_name: The csv file name
    :return: None
    """
    try:
        # @desc: Delete the csv file from the uploaded folder
        os.remove(os.path.join(Directories.CSV_UPLOADED_FOLDER, file_name))
        return jsonify({
            "status": "success",
            "message": f"Analysis completed successfully and {file_name} deleted successfully from the server"
        })
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__))
        return jsonify({"status": "error",
                        "message": "Error in the process of deleting the csv file with the error: " + str(e)}), 500


def professor_analysis(csv_file_path: str):
    """
    evaluatee_list: The list of the professors without duplicates
    evaluatee_department: The department of the professor
    """

    # @desc: In this analysis we are going to automate the account creation of the professors if they don't
    # exist in the database.

    # @desc: Read the csv file
    csv_file = pd.read_csv(csv_file_path)

    # @desc: Get the list of the professors with their department and email address from the csv file and save it in
    # a list of tuples with the format (evaluatee, department, email) and remove the duplicates from the list of
    # tuples
    evaluatee_list = \
        list(set([(row["evaluatee"], row["department"], row["email"])
             for index, row in csv_file.iterrows()]))

    # @desc: Iterate through the list of the professors and check if they exist in the user table of the database
    for index, evaluatee in enumerate(evaluatee_list):
        if not User.query.filter_by(email=evaluatee[2]).first():
            username = evaluatee[2].split("@")[0]
            department = evaluatee[1]
            email = evaluatee[2]
            full_name = evaluatee[0].replace(",", "").title()

            # @desc: Create the user account
            user = User(username=username, email=email,
                        full_name=full_name, department=department, role="user")
            db.session.add(user)
            db.session.commit()
        continue


def course_provider(csv_id: int, csv_file_path: str):
    """
    Get the course provider from the csv file.

    :param csv_id: The csv file id
    :param csv_file_path: The csv file path

    :return: The course provider

    :desc: The course provider scans the csv file and gets the distinct course code and the department of the course
    as well as the names of the professors who taught the course.
    """

    # @desc: Read the csv file
    csv_file = pd.read_csv(csv_file_path)

    # Get the distinct course code and get their department name and the evaluatee name
    course_code_list = list(set([(row["course_code"], row["department"], row["evaluatee"])
                                    for index, row in csv_file.iterrows()]))
    # @desc: Iterate through the course code list and check if the course code exists in the database
    for index, course_code in enumerate(course_code_list):
        if not CsvCourses.query.filter_by(csv_id=csv_id, course_code=course_code[0], course_for_name=course_code[2],
        course_for_department=course_code[1]).first():
            # @desc: Removes the , in the evaluatee name
            course_for_name = course_code[2].replace(",", "")
            csv_course = CsvCourses(csv_id=csv_id, course_code=course_code[0], course_for_name=course_for_name,
            course_for_department=course_code[1])
            db.session.add(csv_course)
            db.session.commit()
        continue


def collection_provider_analysis(csv_id: int, csv_name: str, csv_question: str, csv_file_path: str, school_year: str,
                                 school_semester: str):
    """
    collection_provider_list: The list of the collection providers without duplicates
    collection_provider_overall_sentiment: The overall sentiment of the collection provider
    collection_provider_evaluatee: The number of evaluatee per collection provider
    collection_provider_number_of_sentiments: The number of sentiments of the collection provider
    collection_provider_positive_sentiments_percentage: The positive sentiments percentage of the collection provider
    collection_provider_negative_sentiments_percentage: The negative sentiments percentage of the collection provider
    collection_provider_share: The share of the collection provider in the total responses of the students
    """
    # @desc: Read the path of the csv file
    csv_file = pd.read_csv(csv_file_path)

    evaluatee_names = csv_file["evaluatee"].tolist()
    course_codes = csv_file["course_code"].tolist()

    # # @desc: Main dictionary
    path_to_there_main = Directories.CSV_USER_COLLECTION_OF_SENTIMENT_PER_EVALUATEE_FOLDER + "/" + csv_question + \
        "_" + school_year + "_" + school_semester


def remove_stopwords(response):
    """Remove stopwords from text."""
    response = response.lower()
    response = word_tokenize(response)
    response = [word for word in response if word not in stpwrd]
    response = " ".join(response)
    return response


def csv_evaluator(file_name: str, sentence_index: int, school_semester: str, school_year: str, csv_question: str,
                  model: FileStorage):
    """
    Evaluate the csv file.

    :param file_name: The csv file name
    :param sentence_index: The sentence index
    :param school_semester: The school semester
    :param school_year: The school year
    :param csv_question: The csv question
    :param model: The model
    :return: The evaluated csv file
    """
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    # @desc: Check if the csv file has already been evaluated by csv_question and school_year
    if check_csv_name_exists(csv_question, school_year, school_semester):
        return jsonify({"status": "error", "message": "File already evaluated"}), 409

    # If any error occurs, delete the file
    try:
        # @desc: Format the csv file to the required format: sentence, evaluatee, department and course code.
        # @desc: Get the time of the formatting of the csv file to the required format in seconds and milliseconds
        start_time = time.time()
        start_time_pre_formatter = time.time()
        csv_formatter_to_evaluate(file_name, sentence_index)
        end_time_pre_formatter = time.time()

        # @desc: Read the reformatted csv file and return a pandas dataframe object
        csv_to_pred = pd.read_csv(
            Directories.CSV_REFORMATTED_FOLDER + "/" + file_name)

        # remove the rows that have empty values in the sentence column
        start_time_post_formatter = time.time()
        csv_to_pred = csv_to_pred.dropna(subset=["sentence"])
        # Remove the comma in evaluatee column
        csv_to_pred["evaluatee"] = csv_to_pred["evaluatee"].str.replace(
            ",", "")
        end_time_post_formatter = time.time()

        start_time_tokenizer = time.time()
        tokenizer = pickle.load(
            open(Directories.DEEP_LEARNING_MODEL_FOLDER + "/tokenizer.pickle", "rb"))

        # @desc: Get the sentences from the csv file
        sentences = csv_to_pred["sentence"].to_list()

        # @desc: Lowercase
        sentences = [sentence.lower() for sentence in sentences]

        # @desc: Tokenize the sentences
        tokenized_sentences = tokenizer.texts_to_sequences(sentences)
        end_time_tokenizer = time.time()

        # @desc: Pad the tokenized sentences
        start_time_padding = time.time()
        padded_sentences = pad_sequences(
            tokenized_sentences, maxlen=300, padding='post')
        end_time_padding = time.time()

        # @desc: Load the model
        start_time_model = time.time()
        if model is not None:
            print(f"Model: {model}")
            print(f"Model Summary: {model.summary()}")
            print("Model already loaded")
        end_time_model = time.time()

        # @desc: Predict the sentiment of the sentences
        start_time_prediction = time.time()
        predictions = model.predict(padded_sentences)
        end_time_prediction = time.time()

        # @desc: Get the sentiment of the sentences
        start_time_sentiment = time.time()
        predictions = [round(round(prediction[0], 4) * 100, 2)
                       for prediction in predictions]
        end_time_sentiment = time.time()

        # @desc: Add the predictions to the csv file
        start_time_adding_predictions = time.time()
        csv_to_pred["sentiment"] = predictions

        csv_to_pred['sentiment_converted'] = csv_to_pred['sentiment'].apply(
            lambda x: '1' if x >= 50 else '0')
        csv_to_pred['sentence_remove_stopwords'] = csv_to_pred['sentence'].apply(
            remove_stopwords)
        csv_to_pred['review_len'] = csv_to_pred['sentence_remove_stopwords'].astype(
            str).apply(len)
        csv_to_pred['word_count'] = csv_to_pred['sentence_remove_stopwords'].apply(
            lambda x: len(str(x).split()))
        csv_to_pred['polarity'] = csv_to_pred['sentence_remove_stopwords']. \
            map(lambda response: TextBlob(response).sentiment.polarity)

        end_time_adding_predictions = time.time()

        date_processed = Timezone("Asia/Manila").get_timezone_current_time()

        # @desc: Save the csv file details to the database (csv_name, csv_question, csv_file_path, school_year)
        csv_file = CsvModelDetail(
            csv_question=csv_question, school_year=school_year, school_semester=school_semester)
        db.session.add(csv_file)
        db.session.commit()

        # Add to the database
        start_time_adding_to_db = time.time()
        for index, row in csv_to_pred.iterrows():
            result = CsvAnalyzedSentiment(
                csv_id=csv_file.csv_id,
                evaluatee=row["evaluatee"],
                department=row["department"],
                course_code=row["course_code"],
                sentence=row["sentence"],
                sentiment=row["sentiment"],
                sentiment_converted=row["sentiment_converted"],
                sentence_remove_stopwords=row["sentence_remove_stopwords"],
                review_len=row["review_len"],
                word_count=row["word_count"],
                polarity=row["polarity"],
            )
            db.session.add(result)
        db.session.commit()
        end_time_adding_to_db = time.time()

        # @desc: For analysis purposes
        start_time_analysis_user = time.time()
        professor_analysis(
            Directories.CSV_REFORMATTED_FOLDER + "/" + file_name)
        end_time_analysis_user = time.time()

        course_provider(
            csv_id=csv_file.csv_id,
            csv_file_path=Directories.CSV_REFORMATTED_FOLDER + "/" + file_name
        )

        end_time = time.time()
        # @desc: Get the overall time taken to evaluate the csv file
        overall_time = end_time - start_time
        # @desc: Get the time taken to format the csv file to the required format
        pre_formatter_time = end_time_pre_formatter - start_time_pre_formatter
        # @desc: Get the time taken to remove the rows that have empty values in the sentence column
        post_formatter_time = end_time_post_formatter - start_time_post_formatter
        # @desc: Get the time taken to tokenize the sentences
        tokenizer_time = end_time_tokenizer - start_time_tokenizer
        # @desc: Get the time taken to pad the tokenized sentences
        padding_time = end_time_padding - start_time_padding
        # @desc: Get the time taken to load the model
        model_time = end_time_model - start_time_model
        # @desc: Get the time taken to predict the sentiment of the sentences
        prediction_time = end_time_prediction - start_time_prediction
        # @desc: Get the time taken to get the sentiment of the sentences
        sentiment_time = end_time_sentiment - start_time_sentiment
        # @desc: Get the time taken to add the predictions to the csv file
        adding_predictions_time = end_time_adding_predictions - start_time_adding_predictions
        # @desc: Get the time taken to add the csv file to the database
        adding_to_db_time = end_time_adding_to_db - start_time_adding_to_db
        # @desc: Get the time taken to analyze the csv file for the user
        analysis_user_time = end_time_analysis_user - start_time_analysis_user
        # @desc: Get the time taken to analyze the csv file for the department

        # @desc Save the time taken to evaluate the csv file to the database
        time_data = CsvTimeElapsed(csv_id=csv_file.csv_id, date_processed=date_processed, time_elapsed=overall_time,
                                   pre_formatter_time=pre_formatter_time, post_formatter_time=post_formatter_time,
                                   tokenizer_time=tokenizer_time, padding_time=padding_time, model_time=model_time,
                                   prediction_time=prediction_time, sentiment_time=sentiment_time,
                                   adding_predictions_time=adding_predictions_time,
                                   adding_to_db=adding_to_db_time, analysis_user_time=analysis_user_time)

        db.session.add(time_data)
        db.session.commit()
        # @desc: Delete the reformatted csv file from the reformatted folder
        os.remove(os.path.join(Directories.CSV_REFORMATTED_FOLDER, file_name))
        return jsonify({"status": "success",
                        "message": "CSV file evaluated successfully",
                        "csv_file": "Analyzed_" + csv_question + "_" + school_year + ".csv",
                        "overall_time": overall_time,
                        "pre_formatter_time": pre_formatter_time,
                        "post_formatter_time": post_formatter_time,
                        "tokenizer_time": tokenizer_time,
                        "padding_time": padding_time,
                        "model_time": model_time,
                        "prediction_time": prediction_time,
                        "sentiment_time": sentiment_time,
                        "adding_predictions_time": adding_predictions_time,
                        "adding_to_db_time": adding_to_db_time,
                        "analysis_user_time": analysis_user_time,
                        }), 200
    except Exception as e:
        csv_id = db.session.query.with_entities(
            CsvModelDetail.csv_id).filter_by(
            csv_question=csv_question, school_year=school_year, school_semester=school_semester).first()

        if csv_id is not None:
            db.session.delete(
                CsvAnalyzedSentiment.query.filter_by(csv_id=csv_id).all())
            db.session.delete(
                CsvModelDetail.query.filter_by(csv_id=csv_id).first())
            db.session.commit()

        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__))
        return jsonify({"status": "error",
                        "message": "Error in the process of evaluating the csv file with the error: " + str(e)}), 500


def quad(names=None, sentiment_list=None, type_comp=None, duo_raw=None):
    number_of_sentiments, positive_sentiments_percentage, negative_sentiments_percentage, share, \
        department_evaluatee = [], [], [], [], []
    total = len(sentiment_list)
    for name in names:
        number_of_sentiments.append(
            len([sentiment for sentiment in sentiment_list if sentiment[2] == name])
            if len(sentiment_list) > 0 else 0)

        positive_sentiments_percentage.append(
            round(len([sentiment for sentiment in sentiment_list if sentiment[2] == name and sentiment[3] == 1]) /
                  number_of_sentiments[-1] * 100, 2) if number_of_sentiments[-1] > 0 else 0)

        negative_sentiments_percentage.append(
            round(len([sentiment for sentiment in sentiment_list if sentiment[2] == name and sentiment[3] == 0]) /
                  number_of_sentiments[-1] * 100, 2) if number_of_sentiments[-1] > 0 else 0)

        share.append(
            round(number_of_sentiments[-1] / total * 100, 2) if total > 0 else 0)

        if type_comp == "top_dept":
            department_evaluatee.append(
                db.session.query(User.department).filter(
                    User.department == name, User.role == "user").count()
            )
        if type_comp == "top_prof":
            department_evaluatee.append(duo_raw[names.index(name)][1])

    # Top department with the highest number of positive sentiments percentage
    if type_comp == "top_dept":
        top_department = [
            {
                "id": index,
                "department": department,
                "positive_sentiments_percentage":
                    f"{positive_sentiments_percentage[names.index(department)]}%",
                "negative_sentiments_percentage":
                    f"{negative_sentiments_percentage[names.index(department)]}%",
                "number_of_sentiments": f"{number_of_sentiments[names.index(department)]:,} / {total:,}",
                "share": f"{share[names.index(department)]}%",
                "evaluatee": department_evaluatee[names.index(department)]
            } for index, department in enumerate(
                sorted(names, key=lambda x: positive_sentiments_percentage[names.index(x)],
                       reverse=True), start=0)]

        return top_department
    if type_comp == "top_prof":
        top_professor = [{
            "id": index,
            "professor": professor,
            "positive_sentiments_percentage":
                f"{positive_sentiments_percentage[names.index(professor)]}%",
            "negative_sentiments_percentage":
                f"{negative_sentiments_percentage[names.index(professor)]}%",
            "number_of_sentiments":
                f"{number_of_sentiments[names.index(professor)]:,} / {total:,}",
            "share": f"{share[names.index(professor)]}%",
            "evaluatee_department": department_evaluatee[names.index(professor)]
        } for index, professor in enumerate(
            sorted(names,
                   key=lambda x: positive_sentiments_percentage[names.index(
                       x)],
                   reverse=True),
            start=0)]
        return top_professor
    return None


def read_overall_data_department_analysis_csv_files(school_year: str | None, school_semester: str | None, csv_question: str | None):
    """Count the overall data of the department analysis csv files. This is for the analysis purposes."""

    department_list = db.session.query(User.department).filter(
        User.role == "user").distinct().all()
    departments = [department[0] for department in department_list]
    if school_year is None and school_semester is None and csv_question is None:
        sentiment_list = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.department,
            CsvAnalyzedSentiment.sentiment_converted).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).all()
        top_department = quad(
            names=departments,
            sentiment_list=sentiment_list,
            type_comp="top_dept"
        )
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).distinct().all())

        return jsonify({
            "status": "success",
            "year": f"{starting_year} - {ending_year}",
            "top_department": top_department if len(top_department) > 0 else 0
        }), 200
    if school_year is not None and school_semester is not None and csv_question is not None:
        school_year = InputTextValidation(school_year).to_query_school_year()
        school_semester = InputTextValidation(
            school_semester).to_query_space_under()
        csv_question = InputTextValidation(
            csv_question).to_query_csv_question()
        sentiment_list = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.department,
            CsvAnalyzedSentiment.sentiment_converted).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).filter(
            CsvModelDetail.school_year == school_year, CsvModelDetail.school_semester == school_semester,
            CsvModelDetail.csv_question == csv_question).all()

        top_department = quad(
            names=departments,
            sentiment_list=sentiment_list,
            type_comp="top_dept"
        )
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).all())
        return jsonify({
            "status": "success",
            "year": f"{starting_year} - {ending_year}",
            "top_department": top_department if len(top_department) > 0 else 0
        }), 200


def read_overall_data_professor_analysis_csv_files(school_year: str | None, school_semester: str | None, csv_question: str | None):
    """Count the overall data of the professor analysis csv files. This is for the analysis purposes."""

    user_list = db.session.query(User.full_name, User.department).filter(
        User.role == "user").all()
    users = [user[0].upper() for user in user_list]
    if school_year is None and school_semester is None and csv_question is None:
        sentiment_list = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).all()

        top_professor = quad(
            names=users,
            sentiment_list=sentiment_list,
            type_comp="top_prof",
            duo_raw=user_list
        )
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).distinct().all())

        return jsonify({
            "status": "success",
            "year": f"{starting_year} - {ending_year}",
            "top_professors": top_professor if len(top_professor) > 0 else 0
        }), 200
    if school_year is not None and school_semester is not None and csv_question is not None:
        school_year = InputTextValidation(school_year).to_query_school_year()
        school_semester = InputTextValidation(
            school_semester).to_query_space_under()
        csv_question = InputTextValidation(
            csv_question).to_query_csv_question()
        sentiment_list = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).filter(
            CsvModelDetail.school_year == school_year, CsvModelDetail.school_semester == school_semester,
            CsvModelDetail.csv_question == csv_question).all()

        top_professor = quad(
            names=users,
            sentiment_list=sentiment_list,
            type_comp="top_prof",
            duo_raw=user_list
        )
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).all())
        return jsonify({
            "status": "success",
            "year": f"{starting_year} - {ending_year}",
            "top_professors": top_professor if len(top_professor) > 0 else 0
        }), 200


def options_read_single_data():
    """Options for the read single data route."""
    csv_file = CsvModelDetail.query.all()

    # @desc: Do not return duplicate school_year, school_semester, and csv_question
    school_year = []
    school_semester = []
    csv_question = []

    for csv in csv_file:
        if csv.school_year not in school_year:
            school_year.append(csv.school_year)

        if csv.school_semester not in school_semester:
            school_semester.append(csv.school_semester)

        if csv.csv_question not in csv_question:
            csv_question.append(csv.csv_question)

    # @desc: Make school_year in descending order
    school_year.sort(reverse=True)

    school_year_dict = [
        {
            "id": index,
            "school_year": InputTextValidation(school_year).to_readable_school_year()
        } for index, school_year in enumerate(school_year)
    ]

    school_semester_dict = [
        {
            "id": index,
            "school_semester": InputTextValidation(school_semester).to_readable_school_semester()
        } for index, school_semester in enumerate(school_semester)
    ]

    csv_question_dict = [
        {
            "id": index,
            "csv_question": InputTextValidation(csv_question).to_readable_csv_question()
        } for index, csv_question in enumerate(csv_question)
    ]

    return jsonify({
        "status": "success",
        "school_year": school_year_dict,
        "school_semester": school_semester_dict,
        "csv_question": csv_question_dict
    }), 200


def list_csv_files_to_view_and_delete_pagination(page: int, per_page: int):
    """
    @desc: List all csv files to view, download, and delete in pagination.
    :param page: The page number.
    :param per_page: The number of items per page.
    :return: A list of csv files.
    """
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 401

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    try:
        if user_data.role == "admin":
            csv_files = db.session.query(CsvModelDetail).order_by(
                CsvModelDetail.csv_id.desc()).filter(
                CsvModelDetail.flag_deleted == 0).paginate(page=page, per_page=per_page)
            list_of_csv_files = [
                {
                    "id": csv_file.csv_id,
                    "school_year": InputTextValidation(csv_file.school_year).to_readable_school_year(),
                    "school_semester": InputTextValidation(csv_file.school_semester).to_readable_school_semester(),
                    "csv_question": InputTextValidation(csv_file.csv_question).to_readable_csv_question(),
                    "flag_deleted": csv_file.flag_deleted,
                    "flag_release": csv_file.flag_release,
                } for csv_file in csv_files.items
            ]

            return jsonify({
                "status": "success",
                "csv_files": list_of_csv_files,
                "total_pages": csv_files.pages,
                "current_page": csv_files.page,
                "has_next": csv_files.has_next,
                "has_prev": csv_files.has_prev,
                "next_page": csv_files.next_num,
                "prev_page": csv_files.prev_num,
                "total_items": csv_files.total,
            }), 200
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 403
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to list the csv files."}), 500


def list_csv_files_to_permanently_delete_pagination(page: int, per_page: int):
    """
    @desc: List all csv files to permanently delete in pagination.
    :param page: The page number.
    :param per_page: The number of items per page.
    :return: A list of csv files.
    """
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 401

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    try:
        if user_data.role == "admin":
            csv_files = db.session.query(CsvModelDetail).filter_by(flag_deleted=True).order_by(
                CsvModelDetail.csv_id.desc()).paginate(page=page, per_page=per_page)

            list_of_csv_files = [
                {
                    "id": csv_file.csv_id,
                    "school_year": InputTextValidation(csv_file.school_year).to_readable_school_year(),
                    "school_semester": InputTextValidation(csv_file.school_semester).to_readable_school_semester(),
                    "csv_question": InputTextValidation(csv_file.csv_question).to_readable_csv_question(),
                    "flag_deleted": csv_file.flag_deleted,
                    "flag_release": csv_file.flag_release,
                } for csv_file in csv_files.items
            ]

            return jsonify({
                "status": "success",
                "csv_files": list_of_csv_files,
                "total_pages": csv_files.pages,
                "current_page": csv_files.page,
                "has_next": csv_files.has_next,
                "has_prev": csv_files.has_prev,
                "next_page": csv_files.next_num,
                "prev_page": csv_files.prev_num,
                "total_items": csv_files.total,
            }), 200
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 403
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )


def to_view_selected_csv_file(csv_id: int):
    """
    @desc: To view the selected csv file
    :param csv_id: The id of the csv file
    :return: The csv file
    """
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 401

    # Get only the full_name of the user and role to the database.
    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    try:
        professor_file = CsvProfessorModel.query.filter_by(
            csv_id=csv_id).first()
        if user_data.role == "admin":
            department_file = CsvDepartmentModel.query.filter_by(
                csv_id=csv_id).first()

            if professor_file is None and department_file is None:
                return jsonify({"status": "error", "message": "No csv file found."}), 400

            professor_file = pd.read_csv(professor_file.csv_file_path)
            department_file = pd.read_csv(department_file.csv_file_path)

            return jsonify({
                "status": "success",
                "professor_file": professor_file.to_dict(orient="records"),
                "department_file": department_file.to_dict(orient="records")
            }), 200
        if user_data.role == "professor":
            if professor_file is None:
                return jsonify({"status": "error", "message": "No csv file found."}), 400

            professor_file = pd.read_csv(professor_file.csv_file_path)

            return jsonify({
                "status": "success",
                "professor_file": professor_file.to_dict(orient="records")
            }), 200
        return jsonify({"status": "error", "message": "You are not allowed to view this page."}), 403
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to view the selected csv file."}), 500


def to_delete_selected_csv_file_permanent(csv_id: int):
    """
    @desc: Delete the selected csv file
    :param csv_id: The id of the csv file
    :return: A json response
    """
    try:
        csv_file = db.session.query(CsvModel).filter_by(csv_id=csv_id).first()
        professor_file = db.session.query(
            CsvProfessorModel).filter_by(csv_id=csv_id).first()
        department_file = db.session.query(
            CsvDepartmentModel).filter_by(csv_id=csv_id).first()
        collections_file = db.session.query(
            CsvCollectionModel).filter_by(csv_id=csv_id).first()

        if csv_file is None and professor_file is None and department_file is None and collections_file is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        os.remove(csv_file.csv_file_path)
        os.remove(professor_file.csv_file_path)
        os.remove(department_file.csv_file_path)
        # @desc: Collections has a subdirectory, so we need to remove it first. Then we can remove the collections file.
        shutil.rmtree(collections_file.csv_file_path)

        db.session.delete(csv_file)
        db.session.delete(professor_file)
        db.session.delete(department_file)
        db.session.delete(collections_file)
        db.session.commit()

        return jsonify({"status": "success", "message": "Successfully deleted the selected csv file with id: "
                                                        + str(csv_id) + ". and its related files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to delete the selected csv file."}), 500


def to_delete_all_csv_file_permanent():
    """
    @desc: Delete all csv files
    :return: A json response
    """
    try:
        # @desc: Get all csv files that is flagged as deleted.
        csv_files = CsvModel.query.filter_by(flag_deleted=True).with_entities(
            CsvModel.csv_id, CsvModel.flag_deleted).all()

        if all(csv_file is None for csv_file in csv_files):
            return jsonify({"status": "error", "message": "No files to be deleted."}), 400

        for csv_file in csv_files:
            csv_file = csv_file[0]
            to_delete_selected_csv_file_permanent(csv_file)
        return jsonify({"status": "success", "message": "Successfully deleted all csv files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to delete all csv files."}), 500


def to_delete_selected_csv_file_flagged(csv_id: int):
    """
    @desc: Flag the selected csv file
    :param csv_id: The id of the csv file
    :return: A json response
    """
    try:
        csv_file = db.session.query(
            CsvModelDetail).filter_by(csv_id=csv_id).first()

        if csv_file is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        if csv_file.flag_deleted == 1:
            return jsonify({"status": "error", "message": "File already Deleted."}), 400

        csv_file.flag_deleted = True

        db.session.commit()

        return jsonify({"status": "success", "message": "Successfully deleted the selected csv file with id: "
                                                        + str(csv_id) + ". and its related files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to flag the selected csv file."}), 500


def to_delete_selected_csv_file_unflagged(csv_id: int):
    """
    @desc: Unflag the selected csv file
    :param csv_id: The id of the csv file
    :return: A json response
    """
    try:
        csv_file = db.session.query(
            CsvModelDetail).filter_by(csv_id=csv_id).first()

        if csv_file is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        if csv_file.flag_deleted == 0:
            return jsonify({"status": "error", "message": "File already Restored."}), 400

        csv_file.flag_deleted = False

        db.session.commit()

        return jsonify({"status": "success", "message": "Successfully restored the selected csv file with id: "
                                                        + str(csv_id) + ". and its related files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to unflag the selected csv file."}), 500


def to_delete_all_csv_files_flag():
    """
    @desc: Flag all csv files
    :return: A json response
    """
    try:
        csv_files = db.session.query(CsvModelDetail).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.flag_deleted).all()

        if all(csv_file[1] == 1 for csv_file in csv_files):
            return jsonify({"status": "error", "message": "All files already Deleted."}), 400

        for csv_file in csv_files:
            csv_file = csv_file[0]
            to_delete_selected_csv_file_flagged(csv_file)

        return jsonify({"status": "success", "message": "Successfully flagged all csv files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to flag all csv files."}), 500


def to_delete_all_csv_files_unflag():
    """
    @desc: Unflag all csv files
    :return: A json response
    """
    try:
        csv_files = db.session.query(CsvModelDetail).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.flag_deleted).all()

        if all(csv_file[1] == 0 for csv_file in csv_files):
            return jsonify({"status": "error", "message": "All files already Restored."}), 400

        for csv_file in csv_files:
            csv_file = csv_file[0]
            to_delete_selected_csv_file_unflagged(csv_file)

        return jsonify({"status": "success", "message": "Successfully unflagged all csv files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to unflag all csv files."}), 500


def to_publish_selected_csv_file(csv_id: int):
    """
    @desc: Publish the selected csv file
    :param csv_id: The id of the csv file
    :return: A json response
    """
    try:
        csv_file = db.session.query(
            CsvModelDetail).filter_by(csv_id=csv_id).first()

        if csv_file is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        if csv_file.flag_release == 1:
            return jsonify({"status": "error", "message": "File already Published."}), 400

        csv_file.flag_release = True

        db.session.commit()

        return jsonify({"status": "success", "message": "Successfully published the selected csv file with id: "
                                                        + str(csv_id) + "."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to publish the selected csv file."}), 500


def to_unpublished_selected_csv_file(csv_id: int):
    """
    @desc: Unpublished the selected csv file
    :param csv_id: The id of the csv file
    :return: A json response
    """
    try:
        csv_file = db.session.query(
            CsvModelDetail).filter_by(csv_id=csv_id).first()

        if csv_file is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        if csv_file.flag_release == 0:
            return jsonify({"status": "error", "message": "File already Unpublished."}), 400

        csv_file.flag_release = False

        db.session.commit()

        return jsonify({"status": "success", "message": "Successfully unpublished the selected csv file with id: "
                                                        + str(csv_id) + "."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to unpublished the selected csv file."}), 500


def to_publish_all_csv_files():
    """
    @desc: Publish all csv files
    :return: A json response
    """
    try:
        csv_files = db.session.query(CsvModelDetail).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.flag_release).all()

        if all(csv_file[1] == 1 for csv_file in csv_files):
            return jsonify({"status": "error", "message": "All files already Published."}), 400

        for csv_file in csv_files:
            csv_file = csv_file[0]
            to_publish_selected_csv_file(csv_file)

        return jsonify({"status": "success", "message": "Successfully published all csv files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to publish all csv files."}), 500


def to_unpublished_all_csv_files():
    """
    @desc: Unpublished all csv files
    :return: A json response
    """
    try:
        csv_files = db.session.query(CsvModelDetail).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.flag_release).all()

        if all(csv_file[1] == 0 for csv_file in csv_files):
            return jsonify({"status": "error", "message": "All files already Unpublished."}), 400

        for csv_file in csv_files:
            csv_file = csv_file[0]
            to_unpublished_selected_csv_file(csv_file)

        return jsonify({"status": "success", "message": "Successfully unpublished all csv files."}), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to unpublished all csv files."}), 500


def to_download_selected_csv_file(csv_id: int):
    """
    This function is used to download the selected csv file.
    :param csv_id: The id of the csv file to be downloaded.
    :return: The selected csv file.
    """
    try:
        # csv_file = CsvModelDetail.query.filter_by(csv_id=csv_id).first()
        # professor_file = CsvProfessorModel.query.filter_by(
        #     csv_id=csv_id).first()
        # department_file = CsvDepartmentModel.query.filter_by(
        #     csv_id=csv_id).first()
        # collections_file = CsvCollectionModel.query.filter_by(
        #     csv_id=csv_id).first()
        #
        # if csv_file is None and professor_file is None and department_file is None and collections_file is None:
        #     return jsonify({"status": "error", "message": "No csv file found."}), 400
        #
        # temp_file = BytesIO()
        # with ZipFile(temp_file, "w") as zip_file:
        #     zip_file.write(csv_file.csv_file_path, arcname="Raw-Data.csv")
        #     zip_file.write(professor_file.csv_file_path,
        #                    arcname="Professors-Metrics.csv")
        #     zip_file.write(department_file.csv_file_path,
        #                    arcname="Departments-Metrics.csv")
        #     for root, dirs, files in os.walk(collections_file.csv_file_path):
        #         for file in files:
        #             zip_file.write(os.path.join(
        #                 root, file), arcname=os.path.join(os.path.basename(root), file))
        #
        # temp_file.seek(0)
        # return send_file(
        #     path_or_file=temp_file, as_attachment=True,
        #     download_name=csv_file.csv_question + "_" +
        #     csv_file.school_year + "_" + csv_file.school_semester + ".zip",
        #     mimetype="application/zip",
        # ), 200

        # Now that we migrate to a database, there are two tables to join to get the file.
        # The first table is the csv_model_detail_test table, which contains the csv_id, csv_question, school_year,
        # school_semester, flag_delete and flag_release columns. The second table is the csv_analyzed_sentiment_test
        # table, which contains the csv_id, evaluatee, dapartment, course code, sentence, sentiment, sentiment_converted
        # sentence_remove_stopwords, review_len, word_count and polarity columns.
        # We join the two tables using the csv_id column.
        sentiments = db.session.query(
            CsvModelDetail.csv_id,
            CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee, CsvAnalyzedSentiment.department,
            CsvAnalyzedSentiment.course_code, CsvAnalyzedSentiment.sentence, CsvAnalyzedSentiment.sentiment,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len, CsvAnalyzedSentiment.word_count, CsvAnalyzedSentiment.polarity
        ).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).filter(
                CsvModelDetail.csv_id == csv_id).all()

        # If the csv_id is not found in the database, return an error message.
        if sentiments is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        # Append the two tables into a pandas dataframe and convert it to a csv file. The csv file is then saved in the
        # server. The csv file is then sent to the client. The csv file is then deleted from the server.

        # Append the two tables into a pandas dataframe and convert it to a list of dictionaries
        sentiments = [dict(row) for row in sentiments]

        # Convert the list of dictionaries to a pandas dataframe and convert it to a csv file.

        df = pd.DataFrame(sentiments, columns=['csv_id', 'csv_id', 'evaluatee', 'department', 'course_code', 'sentence',
                          'sentiment', 'sentiment_converted', 'sentence_remove_stopwords', 'review_len', 'word_count', 'polarity'])

        # Create a BytesIO object to store the dataframe.
        temp_file = BytesIO()

        # Write the dataframe into the BytesIO object.
        df.to_csv(temp_file, index=False)

        # Set the cursor to the beginning of the BytesIO object.
        temp_file.seek(0)

        # Return the csv file.
        return send_file(
            path_or_file=temp_file, as_attachment=True,
            download_name=f"evaluated_file_no.{csv_id}.csv",
            mimetype="text/csv",
        ), 200

    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to download the selected csv file."}), 500


def list_csv_file_to_read(csv_id: int, folder_name: str):
    """
    This function is used to list the csv file to read.
    :param csv_id: The id of the csv file.
    :param folder_name: The name of the folder.
    :return: The list of the csv file.
    """
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 401

    # Get only the full_name of the user and role to the database.
    user_data: User = User.query.with_entities(
        User.full_name, User.role).filter_by(user_id=user_id).first()

    # Convert the fullname from Rodriguez Andrea to RODRIGUEZ_ANDREA
    user_fullname: str = user_data.full_name.upper().replace(" ", "_")

    try:
        if user_data.role == "admin":
            main_directory = CsvCollectionModel.query.filter_by(
                csv_id=csv_id).first()
            file_path = os.path.join(main_directory.csv_file_path, folder_name)
            file_list = os.listdir(file_path)
            file_list_to_read = [
                {
                    "id": index,
                    "file_name": file_name,
                    "file_title": file_name.split(".")[0].replace("_", " ").title(),
                    "url_friendly_file_name": file_name.split(".")[0].replace(" ", "_").lower()
                } for index, file_name in enumerate(file_list)
            ]

            return jsonify({
                "status": "success",
                "file_path": file_path,
                "file_list": file_list_to_read,
                "topic": InputTextValidation(main_directory.csv_question).to_readable_csv_question(),
                "school_year": InputTextValidation(main_directory.school_year).to_readable_school_year(),
                "school_semester": InputTextValidation(main_directory.school_semester).to_readable_school_semester()
            }), 200
        if user_data.role == "user" and user_fullname == folder_name:
            # Join to CsvModel to check if its flag_release is True and not deleted.
            main_directory = db.session.query(CsvModel, CsvCollectionModel).join(
                CsvCollectionModel, CsvCollectionModel.csv_id == CsvModel.csv_id).filter(
                CsvModel.csv_id == csv_id, CsvModel.flag_release == 1, CsvModel.flag_deleted == 0).with_entities(
                CsvCollectionModel.csv_file_path, CsvCollectionModel.csv_question,
                CsvCollectionModel.school_year, CsvCollectionModel.school_semester).first()

            # Check if the main_directory.csv_file_path is not None.
            if main_directory is None:
                return jsonify({"status": "success",
                                "file_path": "",
                                "file_list": [],
                                "topic": "Unavailable",
                                "school_year": "S.Y. 0000-0000",
                                "school_semester": "00-0000000"}), 200

            file_path = os.path.join(main_directory.csv_file_path, folder_name)
            file_list = os.listdir(file_path)
            file_list_to_read = [
                {
                    "id": index,
                    "file_name": file_name,
                    "file_title": file_name.split(".")[0].replace("_", " ").title(),
                    "url_friendly_file_name": file_name.split(".")[0].replace(" ", "_").lower()
                } for index, file_name in enumerate(file_list)
            ]

            return jsonify({
                "status": "success",
                "file_path": file_path,
                "file_list": file_list_to_read,
                "topic": InputTextValidation(main_directory.csv_question).to_readable_csv_question(),
                "school_year": InputTextValidation(main_directory.school_year).to_readable_school_year(),
                "school_semester": InputTextValidation(main_directory.school_semester).to_readable_school_semester()
            }), 200
        return jsonify({"status": "error", "message": "You are not authorized to access this file."}), 401
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to view the directory."}), 500


def to_read_csv_file(csv_id: int, folder_name: str, file_name: str):
    """
    This function is used to read the csv file using pandas.
    :param csv_id: The id of the csv file.
    :param folder_name: The name of the folder.
    :param file_name: The name of the file.
    :return: The csv file.
    """
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 401

    # Get only the full_name of the user and role to the database.
    user_data: User = User.query.with_entities(
        User.full_name, User.role).filter_by(user_id=user_id).first()

    # Convert the fullname from Rodriguez Andrea to RODRIGUEZ_ANDREA
    user_fullname: str = user_data.full_name.upper().replace(" ", "_")

    try:
        if user_data.role == "admin":
            main_directory = CsvCollectionModel.query.filter_by(
                csv_id=csv_id).first()
            file_path = os.path.join(main_directory.csv_file_path, folder_name)
            file_path = os.path.join(file_path, file_name)

            df = pd.read_csv(file_path)
            sentiments_list = [{
                "id": index,
                "sentiment": sentiment,
                "sentences": sentences,
            } for index, (sentiment, sentences) in enumerate(zip(df["sentiment"], df["sentence"]))]

            return jsonify({
                "status": "success",
                "sentiments_list": sentiments_list,
            }), 200
        if user_data.role == "user" and user_fullname == folder_name:

            # Join to CsvModel to check if its flag_release is True and not deleted.
            main_directory = db.session.query(CsvModel, CsvCollectionModel).join(
                CsvCollectionModel, CsvCollectionModel.csv_id == CsvModel.csv_id).filter(
                CsvModel.csv_id == csv_id, CsvModel.flag_release == 1, CsvModel.flag_deleted == 0).with_entities(
                CsvCollectionModel.csv_file_path).first()

            # Check if the main_directory.csv_file_path is not None.
            if main_directory is None:
                return jsonify({"status": "success",
                                "sentiments_list": []}), 200

            file_path = os.path.join(main_directory.csv_file_path, folder_name)
            file_path = os.path.join(file_path, file_name)

            df = pd.read_csv(file_path)

            sentiments_list = [{
                "id": index,
                "sentiment": sentiment,
                "sentences": sentences,
            } for index, (sentiment, sentences) in enumerate(zip(df["sentiment"], df["sentence"]))]

            return jsonify({
                "status": "success",
                "sentiments_list": sentiments_list,
            }), 200
        return jsonify({"status": "error",
                        "message": "You are not authorized to view this file."}), 401
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to read the csv file."}), 500


def list_evaluatees_to_create(page: int, per_page: int):
    """
    This function is used to list the evaluatees to create.
    :return: The list of the evaluatees to create.
    """
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 401

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    try:
        if user_data.role == "admin":
            # @desc: Get users where role is user
            users = User.query.filter_by(role="user").paginate(
                page=page, per_page=per_page, error_out=False)

            # @desc: Get users where role is user and is not in the evaluatee table
            evaluatees_to_create = [
                {
                    "id": user.user_id,
                    "full_name": user.full_name,
                    "email": user.email,
                    "username": user.username,
                    "role": user.role,
                    "department_name": user.department,
                    "is_locked": user.flag_locked,
                    "is_active": user.flag_active,
                    "is_deleted": user.flag_deleted,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at,
                } for user in users
            ]

            return jsonify({
                "status": "success",
                "evaluatees_to_create": evaluatees_to_create,
                "total_pages": users.pages,
                "current_page": users.page,
                "has_next": users.has_next,
                "has_prev": users.has_prev,
                "next_page": users.next_num,
                "prev_page": users.prev_num,
                "total_items": users.total,
            }), 200
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to list the evaluatees to create."}), 500


def list_user_collection_of_sentiment_per_evaluatee_csv_files(page: int):
    """
    This function is used to list the user collection of sentiment per evaluatee csv files.
    :return: The list of the user collection of sentiment per evaluatee csv files.
    """
    try:
        user_collection_of_sentiment_per_evaluatee_csv_files = db.session.query(
            CsvModel, CsvCollectionModel).filter(CsvModel.csv_id == CsvCollectionModel.csv_id).with_entities(
            CsvModel.csv_id, CsvModel.school_year, CsvModel.school_semester, CsvModel.csv_question,
            CsvModel.csv_file_path, CsvModel.csv_name, CsvModel.flag_deleted, CsvModel.flag_release).order_by(
            CsvModel.csv_id.desc()).paginate(
            page=page, per_page=10, error_out=False)

        user_collection_of_sentiment_per_evaluatee_csv_files_to_read = [{
            "id": csv_file.csv_id,
            "school_year": InputTextValidation(csv_file.school_year).to_readable_school_year(),
            "school_semester": InputTextValidation(csv_file.school_semester).to_readable_school_semester(),
            "csv_question": InputTextValidation(csv_file.csv_question).to_readable_csv_question(),
            "csv_file_path": csv_file.csv_file_path,
            "csv_file_name": csv_file.csv_name,
            "flag_deleted": csv_file.flag_deleted,
            "flag_release": csv_file.flag_release,
        } for csv_file in user_collection_of_sentiment_per_evaluatee_csv_files.items]

        return jsonify({
            "status": "success",
            "csv_files": user_collection_of_sentiment_per_evaluatee_csv_files_to_read,
            "total_pages": user_collection_of_sentiment_per_evaluatee_csv_files.pages,
            "current_page": user_collection_of_sentiment_per_evaluatee_csv_files.page,
            "has_next": user_collection_of_sentiment_per_evaluatee_csv_files.has_next,
            "has_prev": user_collection_of_sentiment_per_evaluatee_csv_files.has_prev,
            "next_page": user_collection_of_sentiment_per_evaluatee_csv_files.next_num,
            "prev_page": user_collection_of_sentiment_per_evaluatee_csv_files.prev_num,
            "total_items": user_collection_of_sentiment_per_evaluatee_csv_files.total,
        }), 200
    except Exception as e:
        error_handler(
            name_of=f"Cause of error: {e}",
            error_occurred=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                         function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to list the user collection of sentiment per "
                                   "evaluatee csv files."}), 500


def format_names():
    bulk_up = update(CsvCourses).values(course_for_name=func.replace(CsvCourses.course_for_name, ',', ''))
    db.session.execute(bulk_up)
    db.session.commit()
    return jsonify({"status": "success", "message": "Successfully formatted the names."}), 200