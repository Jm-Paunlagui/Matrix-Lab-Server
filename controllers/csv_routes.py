import os

import pandas as pd
from werkzeug.datastructures import FileStorage

from config.configurations import app
import csv
from database_queries.csv_queries import check_csv_name_exists, save_csv
from flask import jsonify, request

from modules.module import AllowedFile


def upload_csv():
    """
    Upload the csv file to the server.
    """

    # @desc: Get the csv file from the request
    if not request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    csv_file = request.files["csv_file"]

    # @desc: Check if the csv file is empty
    if csv_file.filename == "":
        return jsonify({"status": "error", "message": "No file found"}), 400

    # @desc: Check if the csv file is allowed
    if not AllowedFile(csv_file.filename).allowed_file():
        return jsonify({"status": "error", "message": "File not allowed"}), 400

    # @desc: Check if the csv file already exists
    if check_csv_name_exists(csv_file.filename):
        return jsonify({"status": "error", "message": "File already exists"}), 400

    # @desc: Save the csv file
    csv_file.save(os.path.join(app.config["CSV_FOLDER"], AllowedFile(csv_file.filename).secure_filename()))

    # @desc: Check if the csv file follows the required format: sentence, evaluatee, department and course code.
    csv_file_ = pd.read_csv(app.config["CSV_FOLDER"] + "/" + AllowedFile(csv_file.filename).secure_filename())
    csv_columns = csv_file_.columns
    if csv_columns[0] != "sentence" or csv_columns[1] \
            != "evaluatee" or csv_columns[2] != "department" or csv_columns[3] != "course_code":
        # @desc: Delete the csv file if it does not follow the required format
        os.remove(os.path.join(app.config["CSV_FOLDER"], AllowedFile(csv_file.filename).secure_filename()))
        return jsonify({"status": "error", "message": "Invalid csv file format"}), 400


    # if csv_header[0] != "sentence" or csv_header[1] != "evaluatee" or csv_header[2] != "department" \
    #         or csv_header[3] != "course_code":  # noqa: E501
    #     return jsonify({"status": "error", "message": "Invalid csv file format"}), 400

    # csv_file_path = os.path.join(app.config["CSV_FOLDER"], AllowedFile(csv_file.filename).secure_filename())
    #
    # # @desc: Save csv file details to the database
    # if not save_csv(csv_file.filename, csv_file_path):
    #     return jsonify({"status": "error", "message": "File upload failed"}), 400
    #
    # @desc: Save the csv file
    # csv_file.save(os.path.join(app.config["CSV_FOLDER"], AllowedFile(csv_file.filename).secure_filename()))
    #
    # return save_csv(csv_file.filename, csv_file_path)

