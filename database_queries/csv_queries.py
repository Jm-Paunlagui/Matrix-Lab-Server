
from flask import jsonify, Response

from config.configurations import db
from models.csv_model import CsvModel


def check_csv_name_exists(csv_name: str) -> bool:
    """
    Check if the csv name exists in the database.

    :param csv_name: The csv name to be checked
    :return: True if the csv name exists, False otherwise
    """
    csv = CsvModel.query.filter_by(csv_name=csv_name).first()
    return True if csv else False


def save_csv(csv_name: str, csv_file_path: str) -> tuple[Response, int]:
    """
    Save the csv file details to the database.

    :param csv_name: The csv name
    :param csv_file_path: The csv file path
    :return: The status and message
    """
    csv = CsvModel(csv_name=csv_name, csv_file_path=csv_file_path)
    db.session.add(csv)
    db.session.commit()
    return jsonify({"status": "success", "message": "File uploaded successfully"}), 200
