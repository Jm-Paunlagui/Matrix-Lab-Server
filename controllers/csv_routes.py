import os

import pandas as pd
from werkzeug.datastructures import FileStorage

from config.configurations import app
import csv
from database_queries.csv_queries import view_columns_with_pandas, csv_evaluator, get_all_the_details_from_csv, \
    get_top_department
from flask import jsonify, request

from modules.module import AllowedFile, InputTextValidation


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
    school_semester: str = request.json["selected_semester"]
    school_year: str = request.json["school_year"]
    csv_question: str = request.json["csv_question"]

    if not InputTextValidation().validate_empty_fields(csv_file, sentence_column, school_semester, school_year,
                                                       csv_question):
        return jsonify({"status": "error", "message": "Some of the inputs are unsuccessfully retrieved"}), 400
    if not InputTextValidation(sentence_column).validate_number():
        return jsonify({"status": "error", "message": "Invalid column number"}), 400
    if not InputTextValidation(csv_question).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid question"}), 400
    if not InputTextValidation(school_year).validate_school_year():
        return jsonify({"status": "error", "message": "Invalid school year"}), 400
    if not InputTextValidation(school_semester).validate_school_semester():
        return jsonify({"status": "error", "message": "Invalid school semester"}), 400
    return csv_evaluator(csv_file, int(sentence_column), school_semester, school_year, csv_question)


def getting_all_data_from_csv():
    return get_all_the_details_from_csv()


def getting_top_department():
    return get_top_department()

