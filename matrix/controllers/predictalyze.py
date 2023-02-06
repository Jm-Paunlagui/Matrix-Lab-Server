import base64
import inspect
import os
import pickle
import sys
import time
from io import BytesIO
from zipfile import ZipFile

import nltk
import pandas as pd
from flask import jsonify, Response, send_file, request
from keras.models import load_model
from keras.utils import pad_sequences
from nltk import word_tokenize
from sqlalchemy import update, func
from textblob import TextBlob
from werkzeug.datastructures import FileStorage

from config import Directories
from extensions import db
from matrix.controllers.dashboard import core_analysis, get_top_n_words, get_top_n_bigrams, get_top_n_trigrams
from matrix.models.csv_file import CsvModelDetail, CsvAnalyzedSentiment, CsvCourses, CsvProfessorSentiment, \
    CsvDepartmentSentiment, ErrorModel, CsvTimeElapsed
from matrix.models.user import User
from matrix.module import AllowedFile, PayloadSignature, TextPreprocessing, InputTextValidation, error_message, \
    Timezone, get_starting_ending_year, verify_authenticated_token

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


def error_handler(error_type: str, cause_of: str, category_error: str):
    """
    Log the error to the database.

    :param error_type: The error occurred
    :param cause_of: The name of the error
    :param category_error: The category of the error
    """
    db.session.add(ErrorModel(category_error=category_error,
                   error_type=error_type, cause_of=cause_of))
    db.session.commit()
    return jsonify({"status": "error", "message": error_type}), 500


def view_columns_with_pandas(csv_file_to_view: FileStorage) -> tuple[Response, int]:
    """
    View the csv file columns with pandas.

    :param csv_file_to_view: The csv file to view
    :return: The status and message
    """
    try:
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
            "iss": "https://matrix-client.herokuapp.com",
            "sub": "Columns of the csv file",
            "csv_file_name": AllowedFile(csv_file_to_view.filename).secure_filename(),
            "csv_columns": csv_columns_to_return
        }

        csv_columns_token = PayloadSignature(
            payload=csv_columns_payload).encode_payload()

        return jsonify({"status": "success",
                        "message": "File columns viewed successfully",
                        "token_columns": csv_columns_token}), 200
    except Exception as e:
        error_handler(
            category_error="VIEW_COLUMNS",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while viewing the csv file columns",
                        "error": f"{e}"}), 500


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
    try:
        # @desc: Read the csv file and return a pandas dataframe object
        csv_file = pd.read_csv(
            Directories.CSV_UPLOADED_FOLDER + "/" + file_name)

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
    except Exception as e:
        error_handler(
            category_error="CSV_FORMATTER",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while formatting the csv file",
                        "error": f"{e}"}), 500


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
            category_error="FILE_DELETE",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__))
        return jsonify({"status": "error",
                        "message": "Error in the process of deleting the csv file with the error: " + str(e)}), 500


def professor_analysis(csv_file_path: str, csv_id: int):
    """
    evaluatee_list: The list of the professors without duplicates
    evaluatee_department: The department of the professor
    """
    try:
        # @desc: In this analysis we are going to automate the account creation of the professors if they don't
        # exist in the database.
        # @desc: Read the csv file
        csv_file = pd.read_csv(csv_file_path)

        # @desc: Get the list of the professors with their department and email address from the csv file and save it in
        # a list of tuples with the format (evaluatee, department, email) and remove the duplicates from the list of
        # tuples
        evaluatee_list = \
            list({(row["evaluatee"].replace(",", "").title(), row["department"], row["email"])
                  for index, row in csv_file.iterrows()})

        # @desc: Iterate through the list of the professors and check if they exist in the user table of the database
        objects_to_insert = []
        for index, evaluatee in enumerate(evaluatee_list):
            if not User.query.filter_by(full_name=evaluatee[0]).first():
                email = evaluatee[2].lower()
                username = evaluatee[2].split("@")[0]
                department = evaluatee[1]
                full_name = evaluatee[0].replace(",", "").title()

                # @desc: Create the user account
                user = User(username=username, email=email,
                            full_name=full_name, department=department, role="user", verified_email="Verified")
                objects_to_insert.append(user)

        db.session.bulk_save_objects(objects_to_insert)
        db.session.commit()

        # @desc: For each Professor computing code
        sentiment_list = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).filter(
            CsvAnalyzedSentiment.csv_id == csv_id).all()

        user_list = db.session.query(User.full_name, User.department).filter(
            User.role == "user", User.flag_deleted == False).all()

        users = [user[0].upper() for user in user_list]

        quad(
            names=users,
            sentiment_list=sentiment_list,
            type_comp="professor_computing",
            duo_raw=user_list,
            csv_id=csv_id
        )
    except Exception as e:
        error_handler(
            category_error="COMPUTING",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while computing the professors analysis",
                        "error": f"{e}"}), 500


def department_analysis(csv_id: int):
    """department_list: The list of the departments without duplicates"""
    try:
        # @desc: For each Department computing code
        department_list = db.session.query(User.department).filter(
            User.role == "user", User.flag_deleted == False).distinct().all()
        departments = [department[0] for department in department_list]

        sentiment_list = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.department,
            CsvAnalyzedSentiment.sentiment_converted).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id).filter(
            CsvAnalyzedSentiment.csv_id == csv_id).all()

        quad(
            names=departments,
            sentiment_list=sentiment_list,
            type_comp="department_computing",
            csv_id=csv_id
        )
    except Exception as e:
        error_handler(
            category_error="COMPUTING",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while computing the departments analysis",
                        "error": f"{e}"}), 500


def course_provider(csv_id: int, csv_file_path: str):
    """
    Get the course provider from the csv file.

    :param csv_id: The csv file id
    :param csv_file_path: The csv file path

    :return: The course provider

    :desc: The course provider scans the csv file and gets the distinct course code and the department of the course
    as well as the names of the professors who taught the course.
    """
    try:
        # @desc: Read the csv file
        csv_file = pd.read_csv(csv_file_path)

        course_code_sentence_per_professor = csv_file.groupby(["course_code", "evaluatee", "department"]).\
            size().reset_index(name="count")

        courses = []
        for index, row in course_code_sentence_per_professor.iterrows():
            if not CsvCourses.query.filter_by(
                    csv_id=csv_id, course_code=row["course_code"], course_for_name=row["evaluatee"],
                    course_for_department=row["department"], number_of_responses=row["count"]).first():
                # @desc: Removes the , in the evaluatee name
                course_for_name = row["evaluatee"].replace(",", "")
                courses.append(CsvCourses(csv_id=csv_id, course_code=row["course_code"],
                                          course_for_name=course_for_name,
                                          course_for_department=row["department"],
                                          number_of_responses=row["count"]))

        db.session.bulk_save_objects(courses)
        db.session.commit()
    except Exception as e:
        error_handler(
            category_error="CREATE",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while creating the course provider",
                        "error": f"{e}"}), 500


def remove_stopwords(response):
    """Remove stopwords from text."""
    response = response.lower()
    response = word_tokenize(response)
    response = [word for word in response if word not in stpwrd]
    response = " ".join(response)
    return response


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
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    if user_data.role != "admin":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    if user_data.verified_email != "Verified":
        return jsonify({"status": "error", "message": "You are not verified to access this page."}), 401

    previous_evaluated_file = db.session.query(
        CsvModelDetail).with_entities(
        CsvModelDetail.csv_id, CsvModelDetail.school_year, CsvModelDetail.school_semester,
        CsvModelDetail.csv_question,
        CsvModelDetail.flag_deleted, CsvModelDetail.flag_release).order_by(
        CsvModelDetail.csv_id.desc()).filter_by(
        flag_deleted=False).first()

    if previous_evaluated_file:
        # Get the department and professors in the database
        departments_in_db = db.session.query(CsvDepartmentSentiment).with_entities(
            CsvDepartmentSentiment.department).filter_by(
            csv_id=previous_evaluated_file.csv_id).all()
        professors_in_db = db.session.query(CsvProfessorSentiment).with_entities(
            CsvProfessorSentiment.professor).filter_by(
            csv_id=previous_evaluated_file.csv_id).all()

        # Get the department and professors in the csv file
        # @desc: Read the csv file and return a pandas dataframe object
        csv_file = pd.read_csv(
            Directories.CSV_UPLOADED_FOLDER + "/" + file_name)

        # @desc: Get the department and professors in the csv file
        departments_in_csv = csv_file["department"].unique()
        professors_in_csv = csv_file["evaluatee"].unique()

        # Convert the data fetch in the database from [('DAS',), ('DBA',), ('DTE',), ('DCI',)] to
        # ['DAS', 'DBA', 'DTE', 'DCI']
        departments_in_db = [department[0] for department in departments_in_db]
        professors_in_db = [professor[0] for professor in professors_in_db]

        # Check if the departments and professors are the same
        if bool(set(departments_in_db) != set(departments_in_csv) and set(professors_in_db) != set(professors_in_csv)):
            return jsonify({"status": "error",
                            "message": "The csv file does not match the previous evaluated file"}), 400

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
        model = load_model(
            Directories.DEEP_LEARNING_MODEL_FOLDER + "/model.h5")
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
        objects_to_insert = []
        for index, row in csv_to_pred.iterrows():
            objects_to_insert.append(CsvAnalyzedSentiment(
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
            ))
        db.session.bulk_save_objects(objects_to_insert)
        db.session.commit()
        end_time_adding_to_db = time.time()

        # @desc: For analysis purposes
        start_time_analysis_user = time.time()
        professor_analysis(
            csv_file_path=Directories.CSV_REFORMATTED_FOLDER + "/" + file_name, csv_id=csv_file.csv_id)
        end_time_analysis_user = time.time()
        start_time_analysis_department = time.time()
        department_analysis(csv_id=csv_file.csv_id)
        end_time_analysis_department = time.time()
        # @desc: Key provider to the user
        start_time_analysis_collection = time.time()
        course_provider(
            csv_id=csv_file.csv_id,
            csv_file_path=Directories.CSV_REFORMATTED_FOLDER + "/" + file_name
        )
        end_time_analysis_collection = time.time()

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
        analysis_department_time = end_time_analysis_department - \
            start_time_analysis_department
        # @desc: Get the time taken to analyze the csv file for the collection
        analysis_collection_time = end_time_analysis_collection - \
            start_time_analysis_collection

        # @desc Save the time taken to evaluate the csv file to the database
        time_data = CsvTimeElapsed(csv_id=csv_file.csv_id, date_processed=date_processed, time_elapsed=overall_time,
                                   pre_formatter_time=pre_formatter_time, post_formatter_time=post_formatter_time,
                                   tokenizer_time=tokenizer_time, padding_time=padding_time, model_time=model_time,
                                   prediction_time=prediction_time, sentiment_time=sentiment_time,
                                   adding_predictions_time=adding_predictions_time, adding_to_db=adding_to_db_time,
                                   analysis_user_time=analysis_user_time,
                                   analysis_department_time=analysis_department_time,
                                   analysis_collection_time=analysis_collection_time)

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
                        "analysis_department_time": analysis_department_time,
                        "analysis_collection_time": analysis_collection_time
                        }), 200
    except Exception as e:
        csv_id = db.session.query(CsvModelDetail.csv_id).filter_by(
            csv_question=csv_question, school_year=school_year, school_semester=school_semester).first()
        csv_id = csv_id[0]
        if csv_id is not None:
            db.session.query(CsvModelDetail).filter_by(csv_id=csv_id).delete()
            db.session.query(CsvAnalyzedSentiment).filter_by(
                csv_id=csv_id).delete()
            db.session.query(CsvProfessorSentiment).filter_by(
                csv_id=csv_id).delete()
            db.session.query(CsvDepartmentSentiment).filter_by(
                csv_id=csv_id).delete()
            db.session.query(CsvTimeElapsed).filter_by(csv_id=csv_id).delete()
            db.session.commit()
        error_handler(
            category_error="CORE",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__))
        return jsonify({"status": "error",
                        "message": "Error in the process of evaluating the csv file with the error: " + str(e)}), 500


def quad(names=None, sentiment_list=None, type_comp=None, duo_raw=None, csv_id=None):
    try:
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

            if type_comp == "department_computing":
                department_evaluatee.append(
                    db.session.query(User.department).filter(
                        User.department == name, User.role == "user", User.flag_deleted == False).count()
                )
            if type_comp == "professor_computing":
                department_evaluatee.append(duo_raw[names.index(name)][1])

        # Top department with the highest number of positive sentiments percentage
        if type_comp == "professor_computing":
            # Insert the professor's name and the number of sentiments to the database CsvProfessorSentiment
            objects_to_insert = []
            for index, professor in enumerate(
                    sorted(names, key=lambda x: positive_sentiments_percentage[names.index(x)], reverse=True), start=0):
                objects_to_insert.append(CsvProfessorSentiment(
                    csv_id=csv_id,
                    professor=professor,
                    evaluatee_department=department_evaluatee[names.index(
                        professor)],
                    evaluatee_number_of_sentiments=number_of_sentiments[names.index(
                        professor)],
                    evaluatee_positive_sentiments_percentage=positive_sentiments_percentage[names.index(
                        professor)],
                    evaluatee_negative_sentiments_percentage=negative_sentiments_percentage[names.index(
                        professor)],
                    evaluatee_share=share[names.index(professor)],
                ))
            db.session.bulk_save_objects(objects_to_insert)
            db.session.commit()
        if type_comp == "department_computing":
            # Insert the department's name and the number of sentiments to the database CsvDepartmentSentiment
            objects_to_insert = []
            for index, department in enumerate(
                    sorted(names, key=lambda x: positive_sentiments_percentage[names.index(x)], reverse=True), start=0):
                objects_to_insert.append(CsvDepartmentSentiment(
                    csv_id=csv_id,
                    department=department,
                    department_evaluatee=department_evaluatee[names.index(
                        department)],
                    department_number_of_sentiments=number_of_sentiments[names.index(
                        department)],
                    department_positive_sentiments_percentage=positive_sentiments_percentage[names.index(
                        department)],
                    department_negative_sentiments_percentage=negative_sentiments_percentage[names.index(
                        department)],
                    department_share=share[names.index(department)],
                ))
            db.session.bulk_save_objects(objects_to_insert)
            db.session.commit()
        return None
    except Exception as e:
        error_handler(
            category_error="CREATE",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while computing the data in the database.",
                        "error": f"{e}"}), 500


def computed(sentiment_list=None, many=False, type_comp=None, names=None, no_of_evaluated=None, duo_raw=None,
             bulk_download=None):
    """
    name: Professor | Department
    department: Department | Number of Professors
    number_of_sentiments: Number of Sentiments
    positive_sentiments_percentage: positive sentiments percentage float
    negative_sentiments_percentage: negative sentiments percentage float
    share: share of the number of sentiments float
    """
    try:
        # Already computed, therefore return the data from the database
        if not many and type_comp == "NNTC":
            total = [sum([sentiment[4] for sentiment in sentiment_list])][0]
            return [{
                "id": index,
                "name": sentiment[2],
                "positive_sentiments_percentage":
                    f"{sentiment[5]}%",
                "negative_sentiments_percentage":
                    f"{sentiment[6]}%",
                "number_of_sentiments":
                    f"{sentiment[4]:,} / {total}",
                "share": f"{sentiment[7]}%",
                "department": sentiment[3]
            } for index, sentiment in enumerate(sentiment_list, start=0)]

        department_evaluatee, number_of_sentiments, positive_sentiments_percentage, negative_sentiments_percentage, \
            share = [], [], [], [], []
        total = sum([sentiment[4] for sentiment in sentiment_list])
        for name in names:
            number_of_sentiments.append(
                sum([sentiment[4]
                     for sentiment in sentiment_list if sentiment[2] == name])
            )
            # Recalculate the percentage of positive and negative sentiments by department and divide by the number of
            # evaluated files
            positive_sentiments_percentage.append(
                round(
                    sum([sentiment[5] for sentiment in sentiment_list if sentiment[2] == name]) / no_of_evaluated, 2)
            ) if no_of_evaluated > 0 else 0
            negative_sentiments_percentage.append(
                round(
                    sum([sentiment[6] for sentiment in sentiment_list if sentiment[2] == name]) / no_of_evaluated, 2)
            ) if no_of_evaluated > 0 else 0
            share.append(
                round(
                    sum([sentiment[4] for sentiment in sentiment_list if sentiment[2] == name]) / total * 100, 2)
            ) if total > 0 else 0
            if type_comp == "top_dept":
                department_evaluatee.append(
                    db.session.query(User.department).filter(
                        User.department == name, User.role == "user", User.flag_deleted == False).count()
                ) if type_comp == "top_dept" else 0
            if type_comp == "top_prof":
                department_evaluatee.append(
                    duo_raw[names.index(name)][1]) if type_comp == "top_prof" else 0

        top = [
            {
                "id": index,
                "name": name,
                "positive_sentiments_percentage":
                    f"{positive_sentiments_percentage[names.index(name)]}%",
                "negative_sentiments_percentage":
                    f"{negative_sentiments_percentage[names.index(name)]}%",
                "number_of_sentiments": f"{number_of_sentiments[names.index(name)]:,} / {total:,}",
                "share": f"{share[names.index(name)]}%",
                "department": department_evaluatee[names.index(name)]
            } for index, name in enumerate(
                sorted(names, key=lambda x: positive_sentiments_percentage[names.index(x)],
                       reverse=True), start=0)] if positive_sentiments_percentage else []

        if bulk_download is None or False:
            return top

        return [
            (index + 1, index + 1, name, department_evaluatee[names.index(name)], number_of_sentiments[names.index(name)],
                positive_sentiments_percentage[names.index(
                    name)], negative_sentiments_percentage[names.index(name)],
                share[names.index(name)]) for index, name in enumerate(
                    sorted(names, key=lambda x: positive_sentiments_percentage[names.index(x)],
                           reverse=True), start=0)] if positive_sentiments_percentage else []
    except Exception as e:
        error_handler(
            category_error="COMPUTATION",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while computing the already computed data.",
                        "error": f"{e}"}), 500


def read_overall_data_department_analysis_csv_files(school_year: str | None, school_semester: str | None,
                                                    csv_question: str | None):
    """Count the overall data of the department analysis csv files. This is for the analysis purposes."""
    try:
        department_list = db.session.query(User.department).filter(
            User.role == "user", User.flag_deleted == False).distinct().all()
        departments = [department[0] for department in department_list]
        if school_year is None and school_semester is None and csv_question is None:
            no_of_evaluated = db.session.query(CsvModelDetail).filter(
                CsvModelDetail.flag_deleted == False).count()

            sentiment = db.session.query(
                CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
                CsvDepartmentSentiment.department_evaluatee, CsvDepartmentSentiment.department_number_of_sentiments,
                CsvDepartmentSentiment.department_positive_sentiments_percentage,
                CsvDepartmentSentiment.department_negative_sentiments_percentage,
                CsvDepartmentSentiment.department_share).join(
                CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id).filter(
                CsvModelDetail.flag_deleted == False
            ).all()

            top_department = computed(sentiment_list=sentiment, many=True, type_comp="top_dept", names=departments,
                                      no_of_evaluated=no_of_evaluated)

            starting_year, ending_year = get_starting_ending_year(
                db.session.query(CsvModelDetail.school_year).distinct().filter(
                    CsvModelDetail.flag_deleted == False
                ).all())

            return jsonify({
                "status": "success",
                "year": f"{starting_year} - {ending_year}",
                "top_department": top_department if len(top_department) > 0 else 0
            }), 200
        if school_year is not None and school_semester is not None and csv_question is not None:
            school_year = InputTextValidation(
                school_year).to_query_school_year()
            school_semester = InputTextValidation(
                school_semester).to_query_space_under()
            csv_question = InputTextValidation(
                csv_question).to_query_csv_question()

            sentiment_list = db.session.query(
                CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
                CsvDepartmentSentiment.department_evaluatee, CsvDepartmentSentiment.department_number_of_sentiments,
                CsvDepartmentSentiment.department_positive_sentiments_percentage,
                CsvDepartmentSentiment.department_negative_sentiments_percentage,
                CsvDepartmentSentiment.department_share).join(
                CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id).filter(
                CsvModelDetail.school_year == school_year, CsvModelDetail.school_semester == school_semester,
                CsvModelDetail.csv_question == csv_question).all()

            top_department = computed(
                sentiment_list=sentiment_list, type_comp="NNTC")
            starting_year, ending_year = get_starting_ending_year(
                db.session.query(CsvModelDetail.school_year).filter(
                    CsvModelDetail.school_year == school_year).all())
            return jsonify({
                "status": "success",
                "year": f"{starting_year} - {ending_year}",
                "top_department": top_department if len(top_department) > 0 else 0
            }), 200
    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while reading overall data of department analysis csv files.",
                        "error": f"{e}"}), 500


def read_overall_data_professor_analysis_csv_files(school_year: str | None, school_semester: str | None,
                                                   csv_question: str | None):
    """Count the overall data of the professor analysis csv files. This is for the analysis purposes."""
    try:
        user_list = db.session.query(User.full_name, User.department).filter(
            User.role == "user", User.flag_deleted == False).all()
        users = [user[0].upper() for user in user_list]
        if school_year is None and school_semester is None and csv_question is None:
            no_of_evaluated = db.session.query(CsvModelDetail).filter(
                CsvModelDetail.flag_deleted == False).count()

            sentiment = db.session.query(
                CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
                CsvProfessorSentiment.evaluatee_department, CsvProfessorSentiment.evaluatee_number_of_sentiments,
                CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
                CsvProfessorSentiment.evaluatee_negative_sentiments_percentage,
                CsvProfessorSentiment.evaluatee_share).join(
                CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id).filter(
                    CsvModelDetail.flag_deleted == False
            ).all()

            starting_year, ending_year = get_starting_ending_year(
                db.session.query(CsvModelDetail.school_year).distinct().filter(
                    CsvModelDetail.flag_deleted == False
                ).all())

            top_professor = computed(sentiment_list=sentiment, many=True, type_comp="top_prof", names=users,
                                     no_of_evaluated=no_of_evaluated, duo_raw=user_list)

            return jsonify({
                "status": "success",
                "year": f"{starting_year} - {ending_year}",
                "top_professors": top_professor if len(top_professor) > 0 else 0
            }), 200
        if school_year is not None and school_semester is not None and csv_question is not None:
            school_year = InputTextValidation(
                school_year).to_query_school_year()
            school_semester = InputTextValidation(
                school_semester).to_query_space_under()
            csv_question = InputTextValidation(
                csv_question).to_query_csv_question()
            sentiment_list = db.session.query(
                CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
                CsvProfessorSentiment.evaluatee_department, CsvProfessorSentiment.evaluatee_number_of_sentiments,
                CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
                CsvProfessorSentiment.evaluatee_negative_sentiments_percentage,
                CsvProfessorSentiment.evaluatee_share).join(
                CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id).filter(
                CsvModelDetail.school_year == school_year, CsvModelDetail.school_semester == school_semester,
                CsvModelDetail.csv_question == csv_question).all()

            top_professor = computed(
                sentiment_list=sentiment_list, type_comp="NNTC")
            starting_year, ending_year = get_starting_ending_year(
                db.session.query(CsvModelDetail.school_year).filter(
                    CsvModelDetail.school_year == school_year).all())
            return jsonify({
                "status": "success",
                "year": f"{starting_year} - {ending_year}",
                "top_professors": top_professor if len(top_professor) > 0 else 0
            }), 200
    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while reading the overall data of the professor analysis csv files.",
                        "error": f"{e}"}), 500


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
    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    try:
        if user_data.role == "admin" and user_data.verified_email == "Verified":
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
            ] if csv_files.items else []

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
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    try:
        if user_data.role == "admin" and user_data.verified_email == "Verified":
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
            ] if csv_files.items else []

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
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )


def to_view_selected_csv_file(csv_id: int, page: int, per_page: int):
    """
    @desc: To view the selected csv file
    :param csv_id: The id of the csv file
    :param page: The page number.
    :param per_page: The number of items per page.
    :return: The csv file
    """
    # @desc: Get the Session to verify if the user is logged in.
    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    try:
        professor_file = CsvProfessorSentiment.query.filter_by(
            csv_id=csv_id).paginate(page=page, per_page=per_page)
        professor_file_list = [
            {
                "id": index,
                "name": InputTextValidation(professor.professor).to_readable_professor_name(),
                "positive_sentiments_percentage": professor.evaluatee_positive_sentiments_percentage,
                "negative_sentiments_percentage": professor.evaluatee_negative_sentiments_percentage,
                "number_of_sentiments": professor.evaluatee_number_of_sentiments,
                "share": professor.evaluatee_share,
                "department": professor.evaluatee_department,
            } for index, professor in enumerate(professor_file, start=0)
        ]
        if user_data.role == "admin" and user_data.verified_email == "Verified":
            department_file = CsvDepartmentSentiment.query.filter_by(
                csv_id=csv_id).all()

            if professor_file is None and department_file is None:
                return jsonify({"status": "error", "message": "No csv file found."}), 400

            department_file = [
                {
                    "id": index,
                    "name": department.department,
                    "positive_sentiments_percentage": department.department_positive_sentiments_percentage,
                    "negative_sentiments_percentage": department.department_negative_sentiments_percentage,
                    "number_of_sentiments": department.department_number_of_sentiments,
                    "share": department.department_share,
                    "department": department.department_evaluatee
                } for index, department in enumerate(department_file, start=0)
            ] if department_file else []

            return jsonify({
                "status": "success",
                "professor_file": professor_file_list,
                "department_file": department_file,
                "total_pages": professor_file.pages,
                "current_page": professor_file.page,
                "has_next": professor_file.has_next,
                "has_prev": professor_file.has_prev,
                "next_page": professor_file.next_num,
                "prev_page": professor_file.prev_num,
                "total_items": professor_file.total,
            }), 200
        if user_data.role == "professor" and user_data.verified_email == "Verified":
            if professor_file is None:
                return jsonify({"status": "error", "message": "No csv file found."}), 400
            return jsonify({
                "status": "success",
                "professor_file": professor_file if professor_file else [],
                "total_pages": professor_file.pages,
                "current_page": professor_file.page,
                "has_next": professor_file.has_next,
                "has_prev": professor_file.has_prev,
                "next_page": professor_file.next_num,
                "prev_page": professor_file.prev_num,
                "total_items": professor_file.total,
            }), 200
        return jsonify({"status": "error", "message": "You are not allowed to view this page."}), 403
    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
        # Bulk delete the evaulated file in the database
        db.session.query(CsvModelDetail).filter_by(csv_id=csv_id).delete()
        db.session.query(CsvAnalyzedSentiment).filter_by(
            csv_id=csv_id).delete()
        db.session.query(CsvProfessorSentiment).filter_by(
            csv_id=csv_id).delete()
        db.session.query(CsvDepartmentSentiment).filter_by(
            csv_id=csv_id).delete()
        db.session.query(CsvCourses).filter_by(csv_id=csv_id).delete()
        db.session.commit()
        return jsonify({"status": "success", "message": "Successfully deleted the selected csv file with id: "
                                                        + str(csv_id) + ". and its related files."}), 200
    except Exception as e:
        error_handler(
            category_error="DELETE",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
        csv_files = CsvModelDetail.query.filter_by(flag_deleted=True).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.flag_deleted).all()

        if all(csv_file is None for csv_file in csv_files):
            return jsonify({"status": "error", "message": "No files to be deleted."}), 400

        for csv_file in csv_files:
            csv_file = csv_file[0]
            to_delete_selected_csv_file_permanent(csv_file)
        return jsonify({"status": "success", "message": "Successfully deleted all csv files."}), 200
    except Exception as e:
        error_handler(
            category_error="DELETE",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
            return jsonify({"status": "error", "message": "File already Archived."}), 400

        csv_file.flag_deleted = True
        csv_file.flag_release = False

        db.session.commit()
        return jsonify({"status": "success", "message": "Successfully archived the selected csv file with id: "
                                                        + str(csv_id) + ". and its related files."}), 200
    except Exception as e:
        error_handler(
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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

        return jsonify({"status": "success", "message": "Successfully Archived all csv files."}), 200
    except Exception as e:
        error_handler(
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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

        return jsonify({"status": "success", "message": "Successfully restored all csv files."}), 200
    except Exception as e:
        error_handler(
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
            category_error="PUT",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to unpublished all csv files."}), 500


def download_analysis(professors=None, departments=None, courses=None, sentiments=None, analysis=None,
                      type_of_download=None, csv_id=None, file_name=None, bulk_download=False, title=None):
    # Sort form highest to lowest professor positive sentiments percentage and add the rank column to the dataframe.

    professors = sorted(professors, key=lambda x: x[5], reverse=True)

    departments = sorted(departments, key=lambda x: x[5], reverse=True)

    # Sort form highest to lowest course number of responses.
    courses = sorted(courses, key=lambda x: x[5], reverse=True)

    # Append the two tables into a pandas dataframe and convert it to a list of dictionaries
    sentiments = [tuple(row) for row in sentiments]

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
        wordcloud_list_with_sentiment = core_analysis(
            analysis, None, title=title)

    # Convert the list of dictionaries to a pandas dataframe and convert it to a csv file.
    dfraw = pd.DataFrame(
        sentiments, columns=['csv_id', 'csv_id', 'evaluatee', 'department', 'course_code', 'sentence',
                             'sentiment', 'sentiment_converted', 'sentence_remove_stopwords', 'review_len',
                             'word_count', 'polarity'])

    # Convert the list of dictionaries to a pandas dataframe and convert it to a csv file.
    dfprof = pd.DataFrame(
        professors, columns=['csv_id', 'csv_id', 'professor', 'evaluatee_department',
                             'evaluatee_number_of_sentiments', 'evaluatee_positive_sentiments_percentage',
                             'evaluatee_negative_sentiments_percentage', 'evaluatee_share'])

    # Convert the list of dictionaries to a pandas dataframe and convert it to a csv file.
    dfdept = pd.DataFrame(
        departments, columns=['csv_id', 'csv_id', 'department', 'department_evaluatee',
                              'department_number_of_sentiments', 'department_positive_sentiments_percentage',
                              'department_negative_sentiments_percentage', 'department_share'])

    # Convert the list of dictionaries to a pandas dataframe and convert it to a csv file.
    dfcourse = pd.DataFrame(
        courses, columns=['csv_id', 'csv_id', 'course_code', 'course_for_name', 'course_for_department',
                          'number_of_responses'])

    dfunigram = pd.DataFrame(
        get_top_n_words(wordcloud_list_with_sentiment, 30), columns=['id', 'word', 'sentiment', 'frequency'])

    dfbigram = pd.DataFrame(
        get_top_n_bigrams(wordcloud_list_with_sentiment, 30), columns=['id', 'word', 'sentiment', 'frequency'])

    dftrigram = pd.DataFrame(
        get_top_n_trigrams(wordcloud_list_with_sentiment, 30), columns=['id', 'word', 'sentiment', 'frequency'])

    # base64 to image
    sentiment_polarity_encoded = base64.b64decode(
        sentiment_polarity_encoded)
    sentiment_review_length_encoded = base64.b64decode(
        sentiment_review_length_encoded)
    wordcloud_encoded = base64.b64decode(wordcloud_encoded)

    # Create a BytesIO object to store the dataframe.
    temp_file_raw = BytesIO()
    temp_file_professors = BytesIO()
    temp_file_departments = BytesIO()
    temp_file_courses = BytesIO()

    temp_file_unigram = BytesIO()
    temp_file_bigram = BytesIO()
    temp_file_trigram = BytesIO()

    # Save the image temporarily in memory.
    temp_sentiment_polarity_encoded = BytesIO()
    temp_sentiment_review_length_encoded = BytesIO()
    temp_wordcloud_encoded = BytesIO()

    temp_zip_file = BytesIO()

    # Write the dataframe into the BytesIO object.
    if type_of_download == "csv":
        dfraw.to_csv(temp_file_raw, index=False)
        dfprof.to_csv(temp_file_professors, index=False)
        dfdept.to_csv(temp_file_departments, index=False)
        dfcourse.to_csv(temp_file_courses, index=False)

        dfunigram.to_csv(temp_file_unigram, index=False)
        dfbigram.to_csv(temp_file_bigram, index=False)
        dftrigram.to_csv(temp_file_trigram, index=False)
    if type_of_download == "excel":
        dfraw.to_excel(temp_file_raw, index=False)
        dfprof.to_excel(temp_file_professors, index=False)
        dfdept.to_excel(temp_file_departments, index=False)
        dfcourse.to_excel(temp_file_courses, index=False)

        dfunigram.to_excel(temp_file_unigram, index=False)
        dfbigram.to_excel(temp_file_bigram, index=False)
        dftrigram.to_excel(temp_file_trigram, index=False)

    # Write the image into the BytesIO object.
    temp_sentiment_polarity_encoded.write(sentiment_polarity_encoded)
    temp_sentiment_review_length_encoded.write(
        sentiment_review_length_encoded)
    temp_wordcloud_encoded.write(wordcloud_encoded)

    # Set the cursor to the beginning of the BytesIO object.
    temp_file_raw.seek(0)
    temp_file_professors.seek(0)
    temp_file_departments.seek(0)
    temp_file_courses.seek(0)

    temp_file_unigram.seek(0)
    temp_file_bigram.seek(0)
    temp_file_trigram.seek(0)

    temp_sentiment_polarity_encoded.seek(0)
    temp_sentiment_review_length_encoded.seek(0)
    temp_wordcloud_encoded.seek(0)

    # Create a zip file and add the csv file to it.
    with ZipFile(temp_zip_file, "w") as zf:
        if bulk_download is False:
            if type_of_download == "csv" and csv_id is not None:
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_evaluated_raw_file.csv", temp_file_raw.read())
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_top_professors.csv", temp_file_professors.read())
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_top_departments.csv", temp_file_departments.read())
                zf.writestr(f"evaluated_file_no.{csv_id}_distribution_of_courses_per_professors.csv",
                            temp_file_courses.read())
                zf.writestr(f"evaluated_file_no.{csv_id}_common_words_in_unigrams.csv",
                            temp_file_unigram.read())
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_common_words_in_bigrams.csv", temp_file_bigram.read())
                zf.writestr(f"evaluated_file_no.{csv_id}_common_words_in_trigrams.csv",
                            temp_file_trigram.read())
            if type_of_download == "excel" and csv_id is not None:
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_evaluated_raw_file.xlsx", temp_file_raw.read())
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_top_professors.xlsx", temp_file_professors.read())
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_top_departments.xlsx", temp_file_departments.read())
                zf.writestr(f"evaluated_file_no.{csv_id}_distribution_of_courses_per_professors.xlsx",
                            temp_file_courses.read())
                zf.writestr(f"evaluated_file_no.{csv_id}_common_words_in_unigrams.xlsx",
                            temp_file_unigram.read())
                zf.writestr(
                    f"evaluated_file_no.{csv_id}_common_words_in_bigrams.xlsx", temp_file_bigram.read())
                zf.writestr(f"evaluated_file_no.{csv_id}_common_words_in_trigrams.xlsx",
                            temp_file_trigram.read())
            zf.writestr(f"evaluated_image_no.{csv_id}_sentiment_polarity_encoded.png",
                        temp_sentiment_polarity_encoded.read())
            zf.writestr(f"evaluated_image_no.{csv_id}_sentiment_review_length_encoded.png",
                        temp_sentiment_review_length_encoded.read())
            zf.writestr(
                f"evaluated_image_no.{csv_id}_wordcloud_encoded.png", temp_wordcloud_encoded.read())
        if bulk_download is True:
            if type_of_download == "csv" and csv_id is None:
                zf.writestr("all_in_one_file_evaluated_raw_file.csv",
                            temp_file_raw.read())
                zf.writestr("all_in_one_file_top_professors.csv",
                            temp_file_professors.read())
                zf.writestr("all_in_one_file_top_departments.csv",
                            temp_file_departments.read())
                zf.writestr("all_in_one_file_distribution_of_courses_per_professors.csv",
                            temp_file_courses.read())
                zf.writestr("all_in_one_file_common_words_in_unigrams.csv",
                            temp_file_unigram.read())
                zf.writestr(
                    "all_in_one_file_common_words_in_bigrams.csv", temp_file_bigram.read())
                zf.writestr("all_in_one_file_common_words_in_trigrams.csv",
                            temp_file_trigram.read())
            if type_of_download == "excel" and csv_id is None:
                zf.writestr(
                    "all_in_one_file_evaluated_raw_file.xlsx", temp_file_raw.read())
                zf.writestr("all_in_one_file_top_professors.xlsx",
                            temp_file_professors.read())
                zf.writestr("all_in_one_file_top_departments.xlsx",
                            temp_file_departments.read())
                zf.writestr("all_in_one_file_distribution_of_courses_per_professors.xlsx",
                            temp_file_courses.read())
                zf.writestr("all_in_one_file_common_words_in_unigrams.xlsx",
                            temp_file_unigram.read())
                zf.writestr(
                    "all_in_one_file_common_words_in_bigrams.xlsx", temp_file_bigram.read())
                zf.writestr("all_in_one_file_common_words_in_trigrams.xlsx",
                            temp_file_trigram.read())
            zf.writestr("all_in_one_image_sentiment_polarity_encoded.png",
                        temp_sentiment_polarity_encoded.read())
            zf.writestr("all_in_one_image_sentiment_review_length_encoded.png",
                        temp_sentiment_review_length_encoded.read())
            zf.writestr("all_in_one_image_wordcloud_encoded.png",
                        temp_wordcloud_encoded.read())

    # Set the cursor to the beginning of the BytesIO object.
    temp_zip_file.seek(0)

    # Return the csv files to the client.
    return send_file(
        path_or_file=temp_zip_file, as_attachment=True,
        download_name=f"{file_name}.zip",
        mimetype="application/zip",
    ), 200


def to_download_selected_csv_file(csv_id: int, type_of_download: str | None):
    """
    This function is used to download the selected csv file.
    :param csv_id: The id of the csv file to be downloaded.
    :param type_of_download: The type of the file to be downloaded.
    :return: The selected csv file.
    """
    try:
        for_file_name = db.session.query(CsvModelDetail).filter_by(
            csv_id=csv_id).first()

        if for_file_name is None:
            return jsonify({"status": "error", "message": "No csv file found."}), 400

        # For the file name of the csv file
        file_name = AllowedFile(
            f"Evaluated_File_no.{for_file_name.csv_id}_{for_file_name.csv_question}_{for_file_name.school_year}_{for_file_name.school_semester}"
        ).secure_filename()

        sentiments = db.session.query(
            CsvModelDetail.csv_id,
            CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee, CsvAnalyzedSentiment.department,
            CsvAnalyzedSentiment.course_code, CsvAnalyzedSentiment.sentence, CsvAnalyzedSentiment.sentiment,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len, CsvAnalyzedSentiment.word_count, CsvAnalyzedSentiment.polarity
        ).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == csv_id).filter(
            CsvAnalyzedSentiment.csv_id == csv_id).all()

        professors = db.session.query(
            CsvModelDetail.csv_id,
            CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor, CsvProfessorSentiment.evaluatee_department,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage, CsvProfessorSentiment.evaluatee_share
        ).join(
            CsvProfessorSentiment, CsvModelDetail.csv_id == csv_id).filter(
            CsvProfessorSentiment.csv_id == csv_id).all()

        departments = db.session.query(
            CsvModelDetail.csv_id,
            CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_evaluatee,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage, CsvDepartmentSentiment.department_share
        ).join(
            CsvDepartmentSentiment, CsvModelDetail.csv_id == csv_id).filter(
            CsvDepartmentSentiment.csv_id == csv_id).all()

        courses = db.session.query(
            CsvModelDetail.csv_id,
            CsvCourses.csv_id, CsvCourses.course_code, CsvCourses.course_for_name, CsvCourses.course_for_department,
            CsvCourses.number_of_responses
        ).join(
            CsvCourses, CsvModelDetail.csv_id == csv_id).filter(
            CsvCourses.csv_id == csv_id).all()

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == csv_id).filter(
            CsvAnalyzedSentiment.csv_id == csv_id).all()

        # If the csv_id is not found in the database, return an error message.
        if sentiments is None and professors is None and departments is None and courses is None and analysis is None:
            return jsonify({"status": "error", "message": "No Evaluated file found."}), 400

        print("sentiments", sentiments)
        return download_analysis(
            professors=professors, departments=departments, courses=courses, sentiments=sentiments, analysis=analysis,
            type_of_download=type_of_download, csv_id=csv_id, file_name=file_name, bulk_download=False,
            title=f"{InputTextValidation(for_file_name.school_year).to_readable_school_year()} - "
                  f"{InputTextValidation(for_file_name.school_semester).to_readable_school_semester()} for "
                  f"{InputTextValidation(for_file_name.csv_question).to_readable_csv_question()}"
        )

    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to download the selected csv file."}), 500


def to_download_all_csv_files(type_of_download: str | None):
    try:
        file_name = AllowedFile(
            "All_Evaluated_Files_In").secure_filename()

        # Raw Evaluated File
        raw_evaluated_file = db.session.query(
            CsvModelDetail.csv_id,
            CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee, CsvAnalyzedSentiment.department,
            CsvAnalyzedSentiment.course_code, CsvAnalyzedSentiment.sentence, CsvAnalyzedSentiment.sentiment,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len, CsvAnalyzedSentiment.word_count, CsvAnalyzedSentiment.polarity
        ).join(
            CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id
        ).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        # Overall Professor Sentiment
        user_list = db.session.query(User.full_name, User.department).filter(
            User.role == "user", User.flag_deleted == False).all()
        users = [user[0].upper() for user in user_list]
        no_of_evaluated = db.session.query(CsvModelDetail).filter(
            CsvModelDetail.flag_deleted == False
        ).count()

        sentiment_professor = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_department, CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_share).join(
            CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_professor_cal = computed(
            sentiment_list=sentiment_professor, many=True, type_comp="top_prof", names=users,
            no_of_evaluated=no_of_evaluated, duo_raw=user_list, bulk_download=True)

        # Overall Department Sentiment
        department_list = db.session.query(User.department).filter(
            User.role == "user", User.flag_deleted == False).distinct().all()
        departments = [department[0] for department in department_list]

        sentiment_department = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_evaluatee, CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage,
            CsvDepartmentSentiment.department_share).join(
            CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_department_cal = computed(
            sentiment_list=sentiment_department, many=True, type_comp="top_dept", names=departments,
            no_of_evaluated=no_of_evaluated, bulk_download=True)

        courses = (db.session.query(CsvModelDetail.csv_id, CsvCourses.csv_id, CsvCourses.course_code, CsvCourses.course_for_name,
                                    CsvCourses.course_for_department,
                                    db.func.sum(CsvCourses.number_of_responses))
                   .join(CsvModelDetail, CsvCourses.csv_id == CsvModelDetail.csv_id)
                   .group_by(CsvCourses.course_code, CsvCourses.course_for_name,
                             CsvCourses.course_for_department)
                   .filter(
            CsvModelDetail.flag_deleted == False
        ).all()
        )

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id ==
                 CsvAnalyzedSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        # If the csv_id is not found in the database, return an error message.
        if raw_evaluated_file is None and sentiment_professor is None and sentiment_department is None \
                and courses is None and analysis is None:
            return jsonify({"status": "error", "message": "No Evaluated file found."}), 400

        return download_analysis(
            professors=sentiment_professor_cal, departments=sentiment_department_cal, courses=courses,
            sentiments=raw_evaluated_file, analysis=analysis, type_of_download=type_of_download,
            csv_id=None, file_name=file_name, bulk_download=True, title="Overall"
        )

    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to download all csv files."}), 500


def list_csv_file_to_read(csv_id: int, folder_name: str, page: int, per_page: int):
    """
    This function is used to list the csv file to read.
    :param csv_id: The id of the csv file.
    :param folder_name: The name of the folder.
    :param page: The current page
    :param per_page: The number of files to show
    :return: The list of the csv file.
    """
    # @desc: Get the Session to verify if the user is logged in.
    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    # Get only the full_name of the user and role to the database.
    user_data: User = User.query.with_entities(
        User.full_name, User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    # Convert the fullname from Rodriguez Andrea to RODRIGUEZ_ANDREA
    user_fullname: str = user_data.full_name.upper()
    folder_name = folder_name.replace("_", " ").upper()
    try:
        if user_data.role == "admin" and user_data.verified_email == "Verified":
            main_directory = db.session.query(CsvModelDetail.csv_id, CsvCourses.csv_id, CsvModelDetail.csv_question,
                                              CsvModelDetail.school_year, CsvModelDetail.school_semester,
                                              CsvCourses.course_for_name,
                                              CsvCourses.course_code, CsvCourses.number_of_responses).join(
                CsvCourses, CsvModelDetail.csv_id == CsvCourses.csv_id).filter(
                CsvCourses.course_for_name == folder_name,
                CsvCourses.csv_id == csv_id).paginate(page=page, per_page=per_page)

            file_list_to_read = [
                {
                    "id": index,
                    "file_title": file_title[6],
                    "file_name": file_title[6].replace(" ", "_").lower(),
                    "file_path": file_title[6].replace(" ", "_").lower(),
                    "number_of_responses": file_title[7],
                } for index, file_title in enumerate(main_directory)
            ]
            return jsonify({
                "status": "success",
                "file_list": file_list_to_read if len(file_list_to_read) > 0 else [],
                "topic": InputTextValidation(main_directory.items[2][2]).to_readable_csv_question(),
                "school_year": InputTextValidation(main_directory.items[3][3]).to_readable_school_year(),
                "school_semester": InputTextValidation(main_directory.items[4][4]).to_readable_school_semester(),
                "total_pages": main_directory.pages,
                "current_page": main_directory.page,
                "has_next": main_directory.has_next,
                "has_prev": main_directory.has_prev,
                "next_page": main_directory.next_num,
                "prev_page": main_directory.prev_num,
                "total_items": main_directory.total,
            }), 200
        if user_data.role == "user" and user_fullname == folder_name and user_data.verified_email == "Verified":
            # Join to CsvModelDetail to check if its flag_release is True and not deleted.
            main_directory = db.session.query(CsvModelDetail.csv_id, CsvCourses.csv_id, CsvModelDetail.csv_question,
                                              CsvModelDetail.school_year, CsvModelDetail.school_semester,
                                              CsvCourses.course_for_name,
                                              CsvCourses.course_code, CsvCourses.number_of_responses).join(
                CsvCourses, CsvModelDetail.csv_id == CsvCourses.csv_id).filter(
                CsvModelDetail.csv_id == csv_id, CsvModelDetail.flag_release == 1,
                CsvModelDetail.flag_deleted == 0, CsvCourses.course_for_name == user_fullname).\
                paginate(page=page, per_page=per_page)
            # Check if the main_directory.csv_file_path is not None.
            if main_directory is None:
                return jsonify({"status": "success",
                                "file_path": "",
                                "file_list": [],
                                "topic": "Unavailable",
                                "school_year": "S.Y. 0000-0000",
                                "school_semester": "00-0000000"}), 200

            file_list_to_read = [
                {
                    "id": index,
                    "file_title": file_title[6],
                    "file_name": file_title[6].replace(" ", "_").lower(),
                    "file_path": file_title[6].replace(" ", "_").lower(),
                    "number_of_responses": file_title[7],
                } for index, file_title in enumerate(main_directory)
            ]
            return jsonify({
                "status": "success",
                "file_list": file_list_to_read if len(file_list_to_read) > 0 else [],
                "topic": InputTextValidation(main_directory.items[2][2]).to_readable_csv_question(),
                "school_year": InputTextValidation(main_directory.items[3][3]).to_readable_school_year(),
                "school_semester": InputTextValidation(main_directory.items[4][4]).to_readable_school_semester(),
                "total_pages": main_directory.pages,
                "current_page": main_directory.page,
                "has_next": main_directory.has_next,
                "has_prev": main_directory.has_prev,
                "next_page": main_directory.next_num,
                "prev_page": main_directory.prev_num,
                "total_items": main_directory.total,
            }), 200
        return jsonify({"status": "error", "message": "You are not authorized to access this file."}), 401
    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to view the directory."}), 500


def to_read_csv_file(csv_id: int, folder_name: str, file_name: str, page: int, per_page: int):
    """
    This function is used to read the csv file using pandas.
    :param csv_id: The id of the csv file.
    :param folder_name: The name of the folder.
    :param file_name: The name of the file.
    :param page: The page number.
    :param per_page: The number of items per page.
    :return: The csv file.
    """
    # @desc: Get the Session to verify if the user is logged in.
    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    # Get only the full_name of the user and role to the database.
    user_data: User = User.query.with_entities(
        User.full_name, User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    # Convert the fullname from Rodriguez Andrea to RODRIGUEZ_ANDREA
    user_fullname: str = user_data.full_name.upper()
    folder_name = folder_name.replace("_", " ").upper()
    file_name = file_name.replace("_", " ").title()
    try:
        if user_data.role == "admin" and user_data.verified_email == "Verified":
            sentiments = db.session.query(
                CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.course_code,
                CsvAnalyzedSentiment.sentence, CsvAnalyzedSentiment.sentiment) \
                .join(
                CsvAnalyzedSentiment, CsvAnalyzedSentiment.csv_id == CsvModelDetail.csv_id) \
                .filter(
                CsvModelDetail.csv_id == csv_id, CsvAnalyzedSentiment.csv_id == csv_id,
                CsvAnalyzedSentiment.course_code == file_name, CsvAnalyzedSentiment.evaluatee == folder_name
            ).paginate(page=page, per_page=per_page)

            sentiments_list = [{
                "id": index,
                "sentiment": sentence[4],
                "sentences": sentence[3],
            } for index, sentence in enumerate(sentiments)
            ]
            return jsonify({
                "status": "success",
                "sentiments_list": sentiments_list if len(sentiments_list) > 0 else [],
                "total_pages": sentiments.pages,
                "current_page": sentiments.page,
                "has_next": sentiments.has_next,
                "has_prev": sentiments.has_prev,
                "next_page": sentiments.next_num,
                "prev_page": sentiments.prev_num,
                "total_items": sentiments.total,
            }), 200
        if user_data.role == "user" and user_fullname == folder_name and user_data.verified_email == "Verified":
            # Join to CsvModel to check if its flag_release is True and not deleted.
            sentiments = db.session.query(
                CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.course_code,
                CsvAnalyzedSentiment.sentence, CsvAnalyzedSentiment.sentiment).join(
                CsvAnalyzedSentiment, CsvAnalyzedSentiment.csv_id == CsvModelDetail.csv_id).filter(
                CsvModelDetail.csv_id == csv_id, CsvAnalyzedSentiment.csv_id == csv_id,
                CsvAnalyzedSentiment.course_code == file_name, CsvAnalyzedSentiment.evaluatee == folder_name
            ).paginate(page=page, per_page=per_page)

            # Check if the main_directory.csv_file_path is not None.
            if sentiments is None:
                return jsonify({"status": "success",
                                "sentiments_list": []}), 200

            sentiments_list = [{
                "id": index,
                "sentiment": sentence[4],
                "sentences": sentence[3],
            } for index, sentence in enumerate(sentiments)
            ]
            return jsonify({
                "status": "success",
                "sentiments_list": sentiments_list if len(sentiments_list) > 0 else [],
                "total_pages": sentiments.pages,
                "current_page": sentiments.page,
                "has_next": sentiments.has_next,
                "has_prev": sentiments.has_prev,
                "next_page": sentiments.next_num,
                "prev_page": sentiments.prev_num,
                "total_items": sentiments.total,
            }), 200
        return jsonify({"status": "error",
                        "message": "You are not authorized to view this file."}), 401
    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
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
    token: str = request.cookies.get('token')

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    try:
        if user_data.role == "admin" and user_data.verified_email == "Verified":
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
                } for user in sorted(users.items, key=lambda x: x.flag_active == 1, reverse=True)
            ]

            return jsonify({
                "status": "success",
                "evaluatees_to_create": evaluatees_to_create if len(evaluatees_to_create) > 0 else [],
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
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to list the evaluatees to create."}), 500


def list_user_collection_of_sentiment_per_evaluatee_csv_files(page: int, per_page: int):
    """
    This function is used to list the user collection of sentiment per evaluatee csv files.
    :return: The list of the user collection of sentiment per evaluatee csv files.
    """
    try:
        user_collection_of_sentiment_per_evaluatee_csv_files = db.session.query(
            CsvModelDetail).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.school_year, CsvModelDetail.school_semester,
            CsvModelDetail.csv_question,
            CsvModelDetail.flag_deleted, CsvModelDetail.flag_release).order_by(
            CsvModelDetail.csv_id.desc()).paginate(
            page=page, per_page=per_page, error_out=False)

        user_collection_of_sentiment_per_evaluatee_csv_files_to_read = [{
            "id": csv_file.csv_id,
            "school_year": InputTextValidation(csv_file.school_year).to_readable_school_year(),
            "school_semester": InputTextValidation(csv_file.school_semester).to_readable_school_semester(),
            "csv_question": InputTextValidation(csv_file.csv_question).to_readable_csv_question(),
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
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to list the user collection of sentiment per "
                                   "evaluatee csv files."}), 500


def format_names():
    bulk_up = update(CsvCourses).values(
        course_for_name=func.replace(CsvCourses.course_for_name, ',', ''))
    db.session.execute(bulk_up)
    db.session.commit()

    return jsonify({"status": "success", "message": "Successfully formatted the names."}), 200


def get_previous_evaluated_file():
    """
    This function is used to get the previous evaluated file.
    :return: The previous evaluated file.
    """
    try:
        previous_evaluated_file = db.session.query(
            CsvModelDetail).with_entities(
            CsvModelDetail.csv_id, CsvModelDetail.school_year, CsvModelDetail.school_semester,
            CsvModelDetail.csv_question,
            CsvModelDetail.flag_deleted, CsvModelDetail.flag_release).order_by(
            CsvModelDetail.csv_id.desc()).first()

        if previous_evaluated_file:
            return jsonify({
                "status": "success",
                "p_id": previous_evaluated_file.csv_id if previous_evaluated_file else "",
                "p_school_year": InputTextValidation(previous_evaluated_file.school_year).to_readable_school_year() if
                previous_evaluated_file else "",
                "p_school_semester": InputTextValidation(previous_evaluated_file.school_semester).to_readable_school_semester()
                if previous_evaluated_file else "",
                "p_csv_question": InputTextValidation(previous_evaluated_file.csv_question).to_readable_csv_question()
                if previous_evaluated_file else "",
                "p_flag_deleted": "Yes" if previous_evaluated_file.flag_deleted == 1 else "No",
                "p_flag_release": "Yes" if previous_evaluated_file.flag_release == 1 else "No"
            }), 200
        return jsonify({"status": "error", "p_id": "", "p_school_year": "", "p_school_semester": "",
                        "p_csv_question": "", "p_flag_deleted": "", "p_flag_release": ""}), 200
    except Exception as e:
        error_handler(
            category_error="GET",
            cause_of=f"Cause of error: {e}",
            error_type=error_message(error_class=sys.exc_info()[0], line_error=sys.exc_info()[-1].tb_lineno,
                                     function_name=inspect.stack()[0][3], file_name=__name__)
        )
        return jsonify({"status": "error",
                        "message": "An error occurred while trying to get the previous evaluated file."}), 500
