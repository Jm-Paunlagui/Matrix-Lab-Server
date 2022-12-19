from flask import request, jsonify

from database_queries.dashboard_queries import dashboard_data_overall, sentiment_vs_polarity, \
    options_read_single_data_dashboard
from modules.module import InputTextValidation


def options_for_file_data_dashboard():
    """Get the options for department."""
    return options_read_single_data_dashboard()


def getting_all_data_from_csv():
    """Get all the data from the csv file."""
    return dashboard_data_overall()


def for_sentiment_vs_polarity(school_year, school_semester, csv_question):
    """Get the data for sentiment vs polarity."""
    """Get the top professor by file."""
    return sentiment_vs_polarity(school_year, school_semester, csv_question)
