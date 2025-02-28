from flask import request, jsonify, Blueprint

from matrix.controllers.predictalyze import view_columns_with_pandas, csv_evaluator, done_in_csv_evaluation, \
    options_read_single_data, read_overall_data_department_analysis_csv_files, \
    read_overall_data_professor_analysis_csv_files, \
    list_csv_files_to_view_and_delete_pagination, \
    list_csv_files_to_permanently_delete_pagination, list_user_collection_of_sentiment_per_evaluatee_csv_files, \
    to_view_selected_csv_file, to_delete_selected_csv_file_permanent, to_delete_all_csv_file_permanent, \
    to_delete_selected_csv_file_flagged, to_delete_selected_csv_file_unflagged, to_delete_all_csv_files_flag, \
    to_delete_all_csv_files_unflag, to_publish_selected_csv_file, to_unpublished_selected_csv_file, \
    to_publish_all_csv_files, to_unpublished_all_csv_files, to_download_selected_csv_file, list_csv_file_to_read, \
    to_read_csv_file, list_evaluatees_to_create, format_names, get_previous_evaluated_file, to_download_all_csv_files, \
    check_csv_name_exists
from matrix.models.csv_file import CsvModelDetail
from matrix.module import AllowedFile, InputTextValidation, is_school_year_valid

predictalyze = Blueprint("predictalyze", __name__, url_prefix="/data")


@predictalyze.route("/view-columns", methods=["POST"])
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


@predictalyze.route("/analyze-save-csv", methods=["POST"])
def analyze_save_csv():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    csv_file: str = request.json["file_name"]
    sentence_column: str = request.json["selected_column_for_sentence"]
    school_semester: str = request.json["selected_semester"]
    school_year: str = request.json["school_year"]
    csv_question: str = request.json["csv_question"]
    token: str = request.json["token"]

    if not InputTextValidation().validate_empty_fields(csv_file, sentence_column, school_semester, school_year,
                                                       csv_question):
        return jsonify({"status": "error", "message": "Some of the inputs are unsuccessfully retrieved"}), 400
    if not InputTextValidation(sentence_column).validate_number():
        return jsonify({"status": "error", "message": "Invalid column number!"}), 400
    if not InputTextValidation(csv_question).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid question!"}), 400
    if not InputTextValidation(school_year).validate_school_year():
        return jsonify({"status": "error", "message": "Invalid school year!"}), 400
    if not InputTextValidation(school_semester).validate_school_semester():
        return jsonify({"status": "error", "message": "Invalid school semester!"}), 400
    if not is_school_year_valid(school_year):
        return jsonify({"status": "error", "message": "Invalid school year!"}), 400
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()
    if check_csv_name_exists(csv_question, school_year, school_semester):
        return jsonify({"status": "error", "message": "File already evaluated!"}), 409
    previous_semester = CsvModelDetail.query.filter_by \
        (csv_question=csv_question, school_year=school_year).order_by(CsvModelDetail.school_semester.desc()).first()
    if previous_semester:
        previous_semester = previous_semester.school_semester
        if previous_semester == "1st_Semester" and school_semester != "2nd_Semester":
            return jsonify({"status": "error", "message": "Expected 2nd Semester!"}), 409
        elif previous_semester == "2nd_Semester" and school_semester != "3rd_Semester":
            return jsonify({"status": "error", "message": "Expected 3rd Semester!"}), 409
        elif previous_semester == "3rd_Semester" and school_semester != "Summer":
            return jsonify({"status": "error", "message": "Expected Summer!"}), 409
        elif previous_semester == "Summer" and school_semester != "1st_Semester":
            return jsonify({"status": "error", "message": "Expected 1st Semester!"}), 409
    else:
        if school_semester != "1st_Semester":
            return jsonify({"status": "error", "message": "Expected 1st Semester!"}), 409
        return csv_evaluator(csv_file, int(sentence_column), school_semester, school_year, csv_question, token)


@predictalyze.route("/delete-uploaded-csv-file", methods=["POST"])
def delete_uploaded_csv_file():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    csv_file: str = request.json["file_name"]

    if not InputTextValidation(csv_file).validate_empty_fields():
        return jsonify({"status": "error", "message": "Invalid file name"}), 400

    return done_in_csv_evaluation(csv_file)


@predictalyze.route("/options-for-file", methods=["GET"])
def options_for_file_data():
    """Get the options for department."""
    return options_read_single_data()


@predictalyze.route("/get-top-department-overall", methods=["GET"])
def getting_top_department_overall():
    """Get the top department overall."""
    return read_overall_data_department_analysis_csv_files(None, None, None)


@predictalyze.route("/get-top-professor-overall", methods=["GET"])
def getting_top_professor_overall():
    """Get the top professor overall."""
    return read_overall_data_professor_analysis_csv_files(None, None, None)


@predictalyze.route("/get-top-department-by-file", methods=["POST"])
def getting_top_department_by_file():
    """Get the top department by file."""
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
    return read_overall_data_department_analysis_csv_files(school_year, school_semester, csv_question)


@predictalyze.route("/get-top-professor-by-file", methods=["POST"])
def getting_top_professor_by_file():
    """Get the top professor by file."""
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
    return read_overall_data_professor_analysis_csv_files(school_year, school_semester, csv_question)


@predictalyze.route("/list-of-csv-files-to-view/<int:page>/<int:per_page>/<string:token>", methods=["GET"])
def getting_list_of_csv_files(page: int, per_page: int, token: str):
    """Get the list of csv files."""
    return list_csv_files_to_view_and_delete_pagination(page, per_page, token)


@predictalyze.route("/getting-list-of-temporarily-deleted-csv-files/<int:page>/<int:per_page>/<string:token>", methods=["GET"])
def getting_list_of_temporarily_deleted_csv_files(page: int, per_page: int, token: str):
    """Get the list of temporarily deleted csv files."""
    return list_csv_files_to_permanently_delete_pagination(page, per_page, token)


@predictalyze.route("/list-of-csv-files-to-view-collections/<int:page>/<int:per_page>", methods=["GET"])
def getting_collection_of_csv_files(page: int, per_page: int):
    """Get the collection of csv files."""
    return list_user_collection_of_sentiment_per_evaluatee_csv_files(page, per_page)


@predictalyze.route("/view-csv-file/<int:csv_id>/<int:page>/<int:per_page>/<string:token>", methods=["GET"])
def viewing_csv_file(csv_id: int, page: int, per_page: int, token: str):
    """View the csv file."""
    return to_view_selected_csv_file(csv_id=csv_id, page=page, per_page=per_page, token=token)


@predictalyze.route("/delete-csv-file-permanent/<int:csv_id>", methods=["DELETE"])
def deleting_csv_file_permanent(csv_id: int):
    """Delete the csv file."""
    return to_delete_selected_csv_file_permanent(csv_id)


@predictalyze.route("/deleting-all-csv-file-permanent", methods=["DELETE"])
def deleting_all_csv_file_permanent():
    """Delete all the csv files."""
    return to_delete_all_csv_file_permanent()


@predictalyze.route("/delete-csv-file/<int:csv_id>", methods=["PUT"])
def deleting_csv_file_temporary(csv_id: int):
    """Delete the csv file."""
    return to_delete_selected_csv_file_flagged(csv_id)


@predictalyze.route("/unflag-delete-csv-file/<int:csv_id>", methods=["PUT"])
def unflagging_csv_file_deleted(csv_id: int):
    """Unflag the csv file."""
    return to_delete_selected_csv_file_unflagged(csv_id)


@predictalyze.route("/delete-csv-file-all", methods=["PUT"])
def deleting_all_csv_file_temporary():
    """Delete all the csv file."""
    return to_delete_all_csv_files_flag()


@predictalyze.route("/unflag-all-delete-csv-file", methods=["PUT"])
def unflagging_all_csv_file_deleted():
    """Unflag all the csv file."""
    return to_delete_all_csv_files_unflag()


@predictalyze.route("/publish-selected-csv-file/<int:csv_id>", methods=["PUT"])
def publish_selected_csv_file(csv_id: int):
    """Publish the results."""
    return to_publish_selected_csv_file(csv_id)


@predictalyze.route("/unpublished-selected-csv-file/<int:csv_id>", methods=["PUT"])
def unpublished_selected_csv_file(csv_id: int):
    """Unpublished the results."""
    return to_unpublished_selected_csv_file(csv_id)


@predictalyze.route("/publish-all-csv-file", methods=["PUT"])
def publish_all_csv_file():
    """Publish all the results."""
    return to_publish_all_csv_files()


@predictalyze.route("/unpublished-all-csv-file", methods=["PUT"])
def unpublished_all_csv_file():
    """Unpublished all the results."""
    return to_unpublished_all_csv_files()


@predictalyze.route("/download-csv-file/<int:csv_id>/<string:type_of_download>", methods=["GET"])
def downloading_csv_file(csv_id: int, type_of_download: str):
    """Download the csv file."""
    return to_download_selected_csv_file(csv_id, type_of_download)


@predictalyze.route("/download-all-csv-file/<string:type_of_download>", methods=["GET"])
def downloading_all_csv_file(type_of_download: str):
    """Download all the csv file."""
    return to_download_all_csv_files(type_of_download)


@predictalyze.route("/get-list-of-taught-courses/<int:csv_id>/<string:folder_name>/<int:page>/<int:per_page>/<string:token>",
                    methods=["GET"])
def list_of_csv_files_to_view(csv_id: int, folder_name: str, page: int, per_page: int, token: str):
    """Get the list of csv files to view."""
    return list_csv_file_to_read(csv_id, folder_name, page, per_page, token)


@predictalyze.route("/read-data-response/<int:csv_id>/<string:folder_name>/<string:file_name>/<int:page>/<int:per_page>/<string:token>", methods=["GET"])
def reading_csv_file(csv_id: int, folder_name: str, file_name: str, page: int, per_page: int, token: str):
    """Read the csv file."""
    return to_read_csv_file(csv_id, folder_name, file_name, page, per_page, token)


@predictalyze.route("/list-of-users-to-view/<int:page>/<int:per_page>/<string:token>", methods=["GET"])
def getting_list_of_evaluatees(page: int, per_page: int, token: str):
    """Get the list of evaluatees."""
    return list_evaluatees_to_create(page, per_page, token)


@predictalyze.route("/format-names", methods=["GET"])
def formatting_names():
    return format_names()


@predictalyze.route("get-previous-evaluated-file", methods=["GET"])
def getting_previous_evaluated_file():
    return get_previous_evaluated_file()
