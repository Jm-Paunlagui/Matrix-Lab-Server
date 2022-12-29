from flask import Blueprint

from matrix.controllers.dashboard import options_read_single_data_dashboard, analysis_options_admin, \
    analysis_options_user, dashboard_data_csv, dashboard_data_professor

dashboard = Blueprint("dashboard", __name__, url_prefix="/analysis")


@dashboard.route("/options-for-file-data-dashboard", methods=["GET"])
def options_for_file_data_dashboard():
    """Get the options for department."""
    return options_read_single_data_dashboard()


@dashboard.route("/dashboard-data-csv", methods=["GET"])
def getting_all_data_from_csv():
    """Get all the data from the csv file."""
    return dashboard_data_csv()


@dashboard.route("/dashboard-data-user", methods=["GET"])
def getting_all_user():
    """Get all the data from the csv file."""
    return dashboard_data_professor()


@dashboard.route("/sentiment_vs_polarity/<string:school_year>/<string:school_semester>/<string:csv_question>",
                 methods=["GET"])
def for_analysis_options_admin(school_year, school_semester, csv_question):
    """Get the data for sentiment vs polarity."""
    return analysis_options_admin(school_year, school_semester, csv_question)


@dashboard.route("/for_analysis_options_user/<string:school_year>/<string:school_semester>/<string:csv_question>",
                 methods=["GET"])
def for_analysis_options_user(school_year, school_semester, csv_question):
    """Get the data for sentiment vs polarity."""
    return analysis_options_user(school_year, school_semester, csv_question)
