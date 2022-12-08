from database_queries.csv_queries import view_columns_with_pandas, csv_evaluator, \
    read_overall_data_department_analysis_csv_files, read_overall_data_professor_analysis_csv_files, \
    read_single_data_department_analysis_csv_files, read_single_data_professor_analysis_csv_files, \
    options_read_single_data, dashboard_data_overall, list_csv_files_to_view_and_delete_pagination, \
    to_view_selected_csv_file, to_delete_selected_csv_file, to_download_selected_csv_file, list_csv_file_to_read, \
    to_read_csv_file, list_evaluatees_to_create, done_in_csv_evaluation, \
    list_user_collection_of_sentiment_per_evaluatee_csv_files
from testpy.analyze import get_all_the_details_from_csv, \
    get_top_department_overall, get_top_professors_overall, get_top_professors_by_file, get_top_department_by_file
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


def delete_uploaded_csv_file():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    csv_file: str = request.json["file_name"]

    if not InputTextValidation(csv_file).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid file name"}), 400

    return done_in_csv_evaluation(csv_file)


def options_for_file_data():
    """
    Get the options for department.
    """
    return options_read_single_data()


def getting_all_data_from_csv():
    """
    Get all the data from the csv file.
    """
    return dashboard_data_overall()


def getting_top_department_overall():
    """
    Get the top department overall.
    """
    return read_overall_data_department_analysis_csv_files()


def getting_top_professor_overall():
    """
    Get the top professor overall.
    """
    return read_overall_data_professor_analysis_csv_files()


def getting_top_department_by_file():
    """
    Get the top department by file.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    school_year: str = request.json["school_year"]
    school_semester: str = request.json["school_semester"]
    csv_question: str = request.json["csv_question"]

    if not InputTextValidation().validate_empty_fields(school_year, school_semester, csv_question):
        return jsonify({"status": "error", "message": "Some of the inputs are unsuccessfully retrieved"}), 400
    if not InputTextValidation(school_year).validate_school_year():
        return jsonify({"status": "error", "message": "Invalid school year"}), 400
    if not InputTextValidation(school_semester).validate_school_semester():
        return jsonify({"status": "error", "message": "Invalid school semester"}), 400
    if not InputTextValidation(csv_question).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid question"}), 400
    return read_single_data_department_analysis_csv_files(school_year, school_semester, csv_question)


def getting_top_professor_by_file():
    """
    Get the top professor by file.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    school_year: str = request.json["school_year"]
    school_semester: str = request.json["school_semester"]
    csv_question: str = request.json["csv_question"]

    if not InputTextValidation().validate_empty_fields(school_year, school_semester, csv_question):
        return jsonify({"status": "error", "message": "Some of the inputs are unsuccessfully retrieved"}), 400
    if not InputTextValidation(school_year).validate_school_year():
        return jsonify({"status": "error", "message": "Invalid school year"}), 400
    if not InputTextValidation(school_semester).validate_school_semester():
        return jsonify({"status": "error", "message": "Invalid school semester"}), 400
    if not InputTextValidation(csv_question).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid question"}), 400
    return read_single_data_professor_analysis_csv_files(school_year, school_semester, csv_question)


def getting_list_of_csv_files(page: int):
    """
    Get the list of csv files.
    """
    return list_csv_files_to_view_and_delete_pagination(page)


def getting_collection_of_csv_files(page: int):
    """
    Get the collection of csv files.
    """
    return list_user_collection_of_sentiment_per_evaluatee_csv_files(page)


def viewing_csv_file(csv_id: int):
    """
    View the csv file.
    """
    return to_view_selected_csv_file(csv_id)


def deleting_csv_file(csv_id: int):
    """
    Delete the csv file.
    """
    return to_delete_selected_csv_file(csv_id)


def downloading_csv_file(csv_id: int):
    """
    Download the csv file.
    """
    return to_download_selected_csv_file(csv_id)


def list_of_csv_files_to_view(csv_id: int, folder_name: str):
    """
    Get the list of csv files to view.
    """
    return list_csv_file_to_read(csv_id, folder_name)


def reading_csv_file(csv_id: int, folder_name: str, file_name: str):
    """
    Read the csv file.
    """
    return to_read_csv_file(csv_id, folder_name, file_name)


def getting_list_of_evaluatees(page: int):
    """
    Get the list of evaluatees.
    """
    return list_evaluatees_to_create(page)
