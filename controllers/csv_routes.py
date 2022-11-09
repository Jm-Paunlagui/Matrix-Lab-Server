import os

import pandas as pd
from werkzeug.datastructures import FileStorage

from config.configurations import app
import csv
from database_queries.csv_queries import check_csv_name_exists, save_csv, view_columns_with_pandas, csv_evaluator
from flask import jsonify, request

from modules.module import AllowedFile, InputTextValidation


def upload_csv():
    """
    Upload the csv file to the server.

    :return: The status and message
    """

    # @desc: Get the csv file from the request
    if not request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    csv_file = request.files["csv_file"]
    csv_question = request.form["csv_question"]

    # @desc: Check if the csv file is empty
    if csv_file.filename == "":
        return jsonify({"status": "error", "message": "No file found"}), 400

    if not InputTextValidation().validate_empty_fields(csv_question):
        return jsonify({"status": "error", "message": "Please enter a question"}), 400

    # @desc: Check if the csv file is allowed
    if not AllowedFile(csv_file.filename).allowed_file():
        return jsonify({"status": "error", "message": "File not allowed"}), 400

    # @desc: Check if the csv file already exists
    if check_csv_name_exists(csv_file.filename, csv_question):
        return jsonify({"status": "error", "message": "File already exists"}), 409

    # @desc: Save the csv file details to the database
    return save_csv(csv_file.filename, app.config["CSV_FOLDER"] + "/" + AllowedFile(
        csv_file.filename).secure_filename(), csv_question, csv_file)


def view_columns():
    """
    View the csv file columns.

    :return: The status and message
    """
    # @desc: Get the csv file from the request
    if not request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    csv_file_to_view = request.files["csv_file_to_view"]

    # @desc: Check if the csv file is empty
    if csv_file_to_view.filename == "":
        return jsonify({"status": "error", "message": "No file found"}), 400

    # @desc: Check if the csv file is allowed
    if not AllowedFile(csv_file_to_view.filename).allowed_file():
        return jsonify({"status": "error", "message": "File not allowed"}), 400

    return view_columns_with_pandas(csv_file_to_view)


def analyze_save_csv():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    csv_file: str = request.json["file_name"]
    sentence_column: str = request.json["selected_column_for_sentence"]
    evaluatee_column: str = request.json["selected_column_for_evaluatee"]
    department_column: str = request.json["selected_column_for_department"]
    course_code_column: str = request.json["selected_column_for_course_code"]
    csv_question: str = request.json["csv_question"]
    school_year: str = request.json["school_year"]

    if not InputTextValidation().validate_empty_fields(csv_file, sentence_column, evaluatee_column, department_column,
                                                       course_code_column, csv_question, school_year):
        return jsonify({"status": "error", "message": "Some of the inputs are unsuccessfully retrieved"}), 400
    if not InputTextValidation(sentence_column).validate_number() and not \
            InputTextValidation(evaluatee_column).validate_number() and not \
            InputTextValidation(department_column).validate_number() and not \
            InputTextValidation(course_code_column).validate_number():
        return jsonify({"status": "error", "message": "Invalid column number"}), 400
    if not InputTextValidation(csv_question).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid question"}), 400
    if not InputTextValidation(school_year).validate_school_year():
        return jsonify({"status": "error", "message": "Invalid school year"}), 400
    return csv_evaluator(csv_file, int(sentence_column), int(evaluatee_column), int(department_column),
                         int(course_code_column), csv_question, school_year)
