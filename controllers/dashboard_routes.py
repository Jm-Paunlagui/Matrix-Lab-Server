
from database_queries.dashboard_queries import dashboard_data_overall, analysis_options_admin, \
    options_read_single_data_dashboard, analysis_options_user


def options_for_file_data_dashboard():
    """Get the options for department."""
    return options_read_single_data_dashboard()


def getting_all_data_from_csv():
    """Get all the data from the csv file."""
    return dashboard_data_overall()


def for_analysis_options_admin(school_year, school_semester, csv_question):
    """Get the data for sentiment vs polarity."""
    """Get the top professor by file."""
    return analysis_options_admin(school_year, school_semester, csv_question)


def for_analysis_options_user(school_year, school_semester, csv_question):
    """Get the data for sentiment vs polarity."""
    """Get the top professor by file."""
    return analysis_options_user(school_year, school_semester, csv_question)
