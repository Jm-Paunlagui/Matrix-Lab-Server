import os
from typing import Tuple

import pandas as pd
from flask import jsonify, Response
from werkzeug.datastructures import FileStorage

from config.configurations import db, app
from models.csv_model import CsvModel
from modules.module import AllowedFile, PayloadSignature
from json import JSONEncoder


def check_csv_name_exists(csv_name: str, csv_question: str) -> bool:
    """
    Check if the csv name exists in the database.

    :param csv_name: The csv name to be checked
    :param csv_question: The csv question to be checked
    :return: True if the csv name exists, False otherwise
    """
    csv = CsvModel.query.filter_by(
        csv_name=csv_name, csv_question=csv_question).first()
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

    :param csv_file_to_view: The csv file to be viewed
    :return: The status and message
    """

    csv_file_to_view.save(os.path.join(app.config["CSV_REFORMATTED_FOLDER"], AllowedFile(
        csv_file_to_view.filename).secure_filename()))
    csv_file_ = pd.read_csv(
        app.config["CSV_REFORMATTED_FOLDER"] + "/" + AllowedFile(csv_file_to_view.filename).secure_filename())
    csv_columns = csv_file_.columns

    csv_columns_to_return = []
    for column, index_in_column in zip(csv_columns, range(len(csv_columns))):
        csv_columns_to_return.append(
            {"id": index_in_column, "name": column})

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



