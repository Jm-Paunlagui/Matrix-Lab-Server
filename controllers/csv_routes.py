import os

import pandas as pd
from werkzeug.datastructures import FileStorage

from config.configurations import app
import csv
from database_queries.csv_queries import check_csv_name_exists, save_csv, view_columns_with_pandas
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


