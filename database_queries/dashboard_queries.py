import base64
import math
import os
from io import BytesIO

import numpy as np
import pandas as pd
import seaborn as sns
from flask import session, jsonify
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS, WordCloud

from config.configurations import app
from database_queries.csv_queries import get_starting_ending_year
from models.csv_model import CsvProfessorModel, CsvDepartmentModel, CsvModel
from models.user_model import User
from modules.module import InputTextValidation


def options_read_single_data_dashboard():
    """Options for the read single data route."""
    csv_file = CsvModel.query.all()

    # @desc: Do not return duplicate school_year, school_semester, and csv_question
    school_year = []
    school_semester = []
    csv_question = []

    for csv in csv_file:
        if csv.school_year not in school_year:
            school_year.append(csv.school_year)

        if csv.school_semester not in school_semester:
            school_semester.append(csv.school_semester)

        if csv.csv_question not in csv_question:
            csv_question.append(csv.csv_question)

    # @desc: Make school_year in descending order
    school_year.sort(reverse=True)

    school_year_dict = [
        {
            "id": index + 1,
            "school_year": InputTextValidation(school_year).to_readable_school_year()
        } for index, school_year in enumerate(school_year)
    ]

    school_semester_dict = [
        {
            "id": index + 1,
            "school_semester": InputTextValidation(school_semester).to_readable_school_semester()
        } for index, school_semester in enumerate(school_semester)
    ]

    csv_question_dict = [
        {
            "id": index + 1,
            "csv_question": InputTextValidation(csv_question).to_readable_csv_question()
        } for index, csv_question in enumerate(csv_question)
    ]

    # Add a All option for school_year, school_semester, and csv_question for the dropdown
    school_year_dict.insert(0, {"id": 0, "school_year": "All"})
    school_semester_dict.insert(0, {"id": 0, "school_semester": "All"})
    csv_question_dict.insert(0, {"id": 0, "csv_question": "All"})

    return jsonify({
        "status": "success",
        "school_year": school_year_dict,
        "school_semester": school_semester_dict,
        "csv_question": csv_question_dict
    }), 200


def dashboard_data_overall():
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    if user_data.role != "admin":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    # @desc: Read all the csv file in the database for professor
    csv_professor_files = CsvProfessorModel.query.all()

    # @desc: Read all the csv file in the database by accessing the csv_file_path column and get the evaluatee column
    # and return a list of evaluatee
    evaluatee_list = [pd.read_csv(csv_professor_file.csv_file_path)["evaluatee_list"].tolist()
                      for csv_professor_file in csv_professor_files]

    # @desc: Flatten the list of list
    evaluatee_list = [
        evaluatee for evaluatee_list in evaluatee_list for evaluatee in evaluatee_list]

    # @desc: Get the unique evaluatee
    evaluatee_list = list(set(evaluatee_list))

    # @desc: Read all the csv file in the database for department
    csv_department_files = CsvDepartmentModel.query.all()

    # @desc: Read all the csv file in the database by accessing the csv_file_path column and get the department column
    # and return a list of department
    department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                       for csv_department_file in csv_department_files]

    # @desc: Flatten the list of list
    department_list = [
        department for department_list in department_list for department in department_list]

    # @desc: Get the unique department
    department_list = list(set(department_list))

    # @desc: Get the total number of overall_total_course_code and divide it by the total number of files
    # to get the average number of course code per csv file
    department_evaluatee_course_code = [
        pd.read_csv(csv_department_file.csv_file_path)[
            "department_evaluatee_course_code"].tolist()
        for csv_department_file in csv_department_files]

    department_evaluatee_course_code = [
        course_code for course_code_list in department_evaluatee_course_code for course_code in course_code_list]

    department_evaluatee_course_code = sum(
        department_evaluatee_course_code) / len(csv_department_files) if len(csv_department_files) > 0 else 0

    # @desc: Get the total number of csv files in the database
    csv_files = CsvModel.query.all()

    total_csv_files = len(csv_files)

    four_top_details = [
        {"id": 1, "title": "Professors",
         "value": len(evaluatee_list), "icon": "fas fa-user-tie"},
        {"id": 2, "title": "Departments",
         "value": len(department_list), "icon": "fas fa-university"},
        {"id": 3, "title": "Courses",
         "value": round(department_evaluatee_course_code, 2), "icon": "fas fa-book"},
        {"id": 4, "title": "CSV Files",
         "value": total_csv_files, "icon": "fas fa-file-csv"}
    ]

    return jsonify({
        "status": "success", "details": four_top_details, "overall_sentiments": "sentiment_details",
        # "department_sentiments": department_sentiments
    }), 200


def sentiment_vs_polarity(school_year: str, school_semester: str, csv_question: str):
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    if school_year == "All" and school_semester == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.all()

        department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                           for csv_department_file in csv_department_files]

        # @desc: Flatten the list of list
        department_list = [
            department for department_list in department_list for department in department_list]

        actual_department_list = department_list

        # @desc: Get the total number of department_number_of_sentiments
        department_number_of_sentiments = [
            pd.read_csv(csv_department_file.csv_file_path)[
                "department_number_of_sentiments"].tolist()
            for csv_department_file in csv_department_files]

        department_number_of_sentiments = [
            number_of_sentiments
            for number_of_sentiments_list in department_number_of_sentiments
            for number_of_sentiments in number_of_sentiments_list]

        department_number_of_sentiments = sum(department_number_of_sentiments) if len(
            department_number_of_sentiments) > 0 else 0

        department_positive_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_positive_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_negative_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_negative_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_positive_sentiments_percentage = round(
            sum(department_positive_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0
        department_negative_sentiments_percentage = round(
            sum(department_negative_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0

        # @desc: Get the total number of positive sentiments and negative sentiments based on the percentage
        department_positive_sentiments = round(
            department_number_of_sentiments * (department_positive_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_positive_sentiments_percentage else 0

        department_negative_sentiments = round(
            department_number_of_sentiments * (department_negative_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_negative_sentiments_percentage else 0

        starting_year, ending_year = get_starting_ending_year()

        sentiment_details = [
            {"id": 1, "title": "Positive Sentiments",
             "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-smile-beam", "color50": "bg-green-50",
             "color500": "bg-green-500"},
            {"id": 2, "title": "Negative Sentiments",
             "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-frown", "color50": "bg-red-50",
             "color500": "bg-red-500"}
        ]

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.all()

        csv_file_path = [csv_file.csv_file_path for csv_file in csv_files]

        # @dec: Read all the csv files in the database
        csv_files = [pd.read_csv(csv_file) for csv_file in csv_file_path]

        sentiment_converted_list = []
        polarity_list = []
        review_length_list = []
        wordcloud_list = []
        wordcloud_list_positive = []
        wordcloud_list_negative = []

        for csv_file in csv_files:
            sentiment_converted_list.append(
                csv_file["sentiment_converted"].tolist())
            polarity_list.append(csv_file["polarity"].tolist())
            review_length_list.append(csv_file["review_len"].tolist())
            wordcloud_list.append(
                csv_file["sentence_remove_stopwords"].tolist())
            wordcloud_list_positive.append(csv_file[csv_file["sentiment_converted"] == 1]
                                           ["sentence_remove_stopwords"].tolist())
            wordcloud_list_negative.append(csv_file[csv_file["sentiment_converted"] == 0]
                                           ["sentence_remove_stopwords"].tolist())

        sentiment_converted_list = [sentiment for sentiment_list in sentiment_converted_list
                                    for sentiment in sentiment_list]
        polarity_list = [
            polarity for polarity_list in polarity_list for polarity in polarity_list]
        review_length_list = [review_length for review_length_list in review_length_list
                              for review_length in review_length_list]
        wordcloud_list = [
            wordcloud for wordcloud_list in wordcloud_list for wordcloud in wordcloud_list]
        wordcloud_list_positive = [wordcloud for wordcloud_list in wordcloud_list_positive
                                   for wordcloud in wordcloud_list]
        wordcloud_list_negative = [wordcloud for wordcloud_list in wordcloud_list_negative
                                   for wordcloud in wordcloud_list]

        # Plot the graph using matplotlib and seaborn library
        plt.figure(figsize=(8, 5))
        plt.title("Sentiment vs Polarity")
        plt.xlabel("Sentiment")
        plt.ylabel("Polarity")
        sns.set_style("whitegrid")
        sns.boxplot(x=sentiment_converted_list, y=polarity_list)

        # Save the figure to a BytesIO object
        buf_sentiment_polarity = BytesIO()
        plt.savefig(buf_sentiment_polarity, format="png", bbox_inches="tight")
        buf_sentiment_polarity.seek(0)

        # Encode the figure to a base64-encoded string
        sentiment_polarity = buf_sentiment_polarity.getvalue()
        sentiment_polarity_encoded = base64.b64encode(
            sentiment_polarity).decode("utf-8")

        # `matplotlib.pyplot.close()`. This is necessary to prevent memory leaks.
        plt.close()

        # @desc: Get the Sentiment vs Review Length of the csv files using matplotlib and seaborn library
        plt.figure(figsize=(8, 5))
        plt.title("Sentiment vs Review Length")
        plt.xlabel("Sentiment")
        plt.ylabel("Review Length")
        sns.set_style("whitegrid")
        sns.pointplot(x=sentiment_converted_list, y=review_length_list)

        # Save the figure to a BytesIO object
        buf_sentiment_review_length = BytesIO()
        plt.savefig(buf_sentiment_review_length,
                    format="png", bbox_inches="tight")
        buf_sentiment_review_length.seek(0)

        # Encode the figure to a base64-encoded string
        sentiment_review_length = buf_sentiment_review_length.getvalue()
        sentiment_review_length_encoded = base64.b64encode(
            sentiment_review_length).decode("utf-8")

        # `matplotlib.pyplot.close()`. This is necessary to prevent memory leaks.
        plt.close()

        # @desc: Remove the float found in the wordcloud_list
        wordcloud_list = [str(wordcloud) for wordcloud in wordcloud_list]

        # @desc: Get the overall word cloud 540 × 338
        wordcloud = WordCloud(width=540, height=338, random_state=21, max_font_size=110,
                              background_color="white").generate(" ".join(wordcloud_list))

        # Save the figure to a BytesIO object
        buf_wordcloud = BytesIO()
        plt.figure(figsize=(8, 5))
        plt.tight_layout(pad=0)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buf_wordcloud, format="png", bbox_inches="tight")
        buf_wordcloud.seek(0)

        # Encode the figure to a base64-encoded string
        wordcloud = buf_wordcloud.getvalue()
        wordcloud_encoded = base64.b64encode(wordcloud).decode("utf-8")

        # `matplotlib.pyplot.close()`. This is necessary to prevent memory leaks.
        plt.close()

        # @desc: Remove the float found in the wordcloud_list_positive
        wordcloud_list_positive = [str(wordcloud)
                                   for wordcloud in wordcloud_list_positive]

        # @desc: Get the positive word cloud 540 × 338
        wordcloud_positive = WordCloud(width=540, height=338, random_state=21, max_font_size=110,
                                       background_color="white").generate(" ".join(wordcloud_list_positive))

        # Save the figure to a BytesIO object
        buf_wordcloud_positive = BytesIO()
        plt.figure(figsize=(8, 5))
        plt.tight_layout(pad=0)
        plt.imshow(wordcloud_positive, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buf_wordcloud_positive, format="png", bbox_inches="tight")
        buf_wordcloud_positive.seek(0)

        # Encode the figure to a base64-encoded string
        wordcloud_positive = buf_wordcloud_positive.getvalue()
        wordcloud_positive_encoded = base64.b64encode(
            wordcloud_positive).decode("utf-8")

        # `matplotlib.pyplot.close()`. This is necessary to prevent memory leaks.
        plt.close()

        # @desc: Remove the float found in the wordcloud_list_negative
        wordcloud_list_negative = [str(wordcloud)
                                   for wordcloud in wordcloud_list_negative]

        # @desc: Get the negative word cloud 540 × 338
        wordcloud_negative = WordCloud(width=540, height=338, random_state=21, max_font_size=110,
                                       background_color="white").generate(" ".join(wordcloud_list_negative))

        # Save the figure to a BytesIO object
        buf_wordcloud_negative = BytesIO()
        plt.figure(figsize=(8, 5))
        plt.tight_layout(pad=0)
        plt.imshow(wordcloud_negative, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buf_wordcloud_negative, format="png", bbox_inches="tight")
        buf_wordcloud_negative.seek(0)

        # Encode the figure to a base64-encoded string
        wordcloud_negative = buf_wordcloud_negative.getvalue()
        wordcloud_negative_encoded = base64.b64encode(
            wordcloud_negative).decode("utf-8")

        # `matplotlib.pyplot.close()`. This is necessary to prevent memory leaks.
        plt.close()

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "image_path_wordcloud_positive": wordcloud_positive_encoded,
                        "image_path_wordcloud_negative": wordcloud_negative_encoded
                        }), 200

    elif school_year == "All" and school_semester == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(
            csv_question=csv_question).all()

        department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                           for csv_department_file in csv_department_files]

        # @desc: Flatten the list of list
        department_list = [
            department for department_list in department_list for department in department_list]

        actual_department_list = department_list

        # @desc: Get the total number of department_number_of_sentiments
        department_number_of_sentiments = [
            pd.read_csv(csv_department_file.csv_file_path)[
                "department_number_of_sentiments"].tolist()
            for csv_department_file in csv_department_files]

        department_number_of_sentiments = [
            number_of_sentiments
            for number_of_sentiments_list in department_number_of_sentiments
            for number_of_sentiments in number_of_sentiments_list]

        department_number_of_sentiments = sum(department_number_of_sentiments) if len(
            department_number_of_sentiments) > 0 else 0

        department_positive_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_positive_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_negative_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_negative_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_positive_sentiments_percentage = round(
            sum(department_positive_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0
        department_negative_sentiments_percentage = round(
            sum(department_negative_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0

        # @desc: Get the total number of positive sentiments and negative sentiments based on the percentage
        department_positive_sentiments = round(
            department_number_of_sentiments * (department_positive_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_positive_sentiments_percentage else 0

        department_negative_sentiments = round(
            department_number_of_sentiments * (department_negative_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_negative_sentiments_percentage else 0

        starting_year, ending_year = get_starting_ending_year()

        sentiment_details = [
            {"id": 1, "title": "Positive Sentiments",
             "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-smile-beam", "color50": "bg-green-50",
             "color500": "bg-green-500"},
            {"id": 2, "title": "Negative Sentiments",
             "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-frown", "color50": "bg-red-50",
             "color500": "bg-red-500"}
        ]

        return jsonify({"status": "success", "overall_sentiments": sentiment_details}), 200

    elif school_year == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(
            school_semester=school_semester).all()

        department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                           for csv_department_file in csv_department_files]

        # @desc: Flatten the list of list
        department_list = [
            department for department_list in department_list for department in department_list]

        actual_department_list = department_list

        # @desc: Get the total number of department_number_of_sentiments
        department_number_of_sentiments = [
            pd.read_csv(csv_department_file.csv_file_path)[
                "department_number_of_sentiments"].tolist()
            for csv_department_file in csv_department_files]

        department_number_of_sentiments = [
            number_of_sentiments
            for number_of_sentiments_list in department_number_of_sentiments
            for number_of_sentiments in number_of_sentiments_list]

        department_number_of_sentiments = sum(department_number_of_sentiments) if len(
            department_number_of_sentiments) > 0 else 0

        department_positive_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_positive_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_negative_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_negative_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_positive_sentiments_percentage = round(
            sum(department_positive_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0
        department_negative_sentiments_percentage = round(
            sum(department_negative_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0

        # @desc: Get the total number of positive sentiments and negative sentiments based on the percentage
        department_positive_sentiments = round(
            department_number_of_sentiments * (department_positive_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_positive_sentiments_percentage else 0

        department_negative_sentiments = round(
            department_number_of_sentiments * (department_negative_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_negative_sentiments_percentage else 0

        starting_year, ending_year = get_starting_ending_year()

        sentiment_details = [
            {"id": 1, "title": "Positive Sentiments",
             "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-smile-beam", "color50": "bg-green-50",
             "color500": "bg-green-500"},
            {"id": 2, "title": "Negative Sentiments",
             "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-frown", "color50": "bg-red-50",
             "color500": "bg-red-500"}
        ]

        return jsonify({"status": "success", "overall_sentiments": sentiment_details}), 200

    elif school_semester == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(
            school_year=school_year).all()

        department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                           for csv_department_file in csv_department_files]

        # @desc: Flatten the list of list
        department_list = [
            department for department_list in department_list for department in department_list]

        actual_department_list = department_list

        # @desc: Get the total number of department_number_of_sentiments
        department_number_of_sentiments = [
            pd.read_csv(csv_department_file.csv_file_path)[
                "department_number_of_sentiments"].tolist()
            for csv_department_file in csv_department_files]

        department_number_of_sentiments = [
            number_of_sentiments
            for number_of_sentiments_list in department_number_of_sentiments
            for number_of_sentiments in number_of_sentiments_list]

        department_number_of_sentiments = sum(department_number_of_sentiments) if len(
            department_number_of_sentiments) > 0 else 0

        department_positive_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_positive_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_negative_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_negative_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_positive_sentiments_percentage = round(
            sum(department_positive_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0
        department_negative_sentiments_percentage = round(
            sum(department_negative_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0

        # @desc: Get the total number of positive sentiments and negative sentiments based on the percentage
        department_positive_sentiments = round(
            department_number_of_sentiments * (department_positive_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_positive_sentiments_percentage else 0

        department_negative_sentiments = round(
            department_number_of_sentiments * (department_negative_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_negative_sentiments_percentage else 0

        school_year = school_year.replace("S.Y.", "")
        starting_year = school_year.split("-")[0]
        ending_year = school_year.split("-")[1]

        sentiment_details = [
            {"id": 1, "title": "Positive Sentiments",
             "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-smile-beam", "color50": "bg-green-50",
             "color500": "bg-green-500"},
            {"id": 2, "title": "Negative Sentiments",
             "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-frown", "color50": "bg-red-50",
             "color500": "bg-red-500"}
        ]

        return jsonify({"status": "success", "overall_sentiments": sentiment_details}), 200

    elif school_year == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(csv_question=csv_question,
                                                                  school_semester=school_semester).all()

        department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                           for csv_department_file in csv_department_files]

        # @desc: Flatten the list of list
        department_list = [
            department for department_list in department_list for department in department_list]

        actual_department_list = department_list

        # @desc: Get the total number of department_number_of_sentiments
        department_number_of_sentiments = [
            pd.read_csv(csv_department_file.csv_file_path)[
                "department_number_of_sentiments"].tolist()
            for csv_department_file in csv_department_files]

        department_number_of_sentiments = [
            number_of_sentiments
            for number_of_sentiments_list in department_number_of_sentiments
            for number_of_sentiments in number_of_sentiments_list]

        department_number_of_sentiments = sum(department_number_of_sentiments) if len(
            department_number_of_sentiments) > 0 else 0

        department_positive_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_positive_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_negative_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_negative_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_positive_sentiments_percentage = round(
            sum(department_positive_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0
        department_negative_sentiments_percentage = round(
            sum(department_negative_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0

        # @desc: Get the total number of positive sentiments and negative sentiments based on the percentage
        department_positive_sentiments = round(
            department_number_of_sentiments * (department_positive_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_positive_sentiments_percentage else 0

        department_negative_sentiments = round(
            department_number_of_sentiments * (department_negative_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_negative_sentiments_percentage else 0

        starting_year, ending_year = get_starting_ending_year()

        sentiment_details = [
            {"id": 1, "title": "Positive Sentiments",
             "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-smile-beam", "color50": "bg-green-50",
             "color500": "bg-green-500"},
            {"id": 2, "title": "Negative Sentiments",
             "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-frown", "color50": "bg-red-50",
             "color500": "bg-red-500"}
        ]

        return jsonify({"status": "success", "overall_sentiments": sentiment_details}), 200

    # @desc: For the specific school year, school semester, and csv question
    else:
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(csv_question=csv_question,
                                                                  school_semester=school_semester,
                                                                  school_year=school_year).all()

        department_list = [pd.read_csv(csv_department_file.csv_file_path)["department_list"].tolist()
                           for csv_department_file in csv_department_files]

        # @desc: Flatten the list of list
        department_list = [
            department for department_list in department_list for department in department_list]

        actual_department_list = department_list

        # @desc: Get the total number of department_number_of_sentiments
        department_number_of_sentiments = [
            pd.read_csv(csv_department_file.csv_file_path)[
                "department_number_of_sentiments"].tolist()
            for csv_department_file in csv_department_files]

        department_number_of_sentiments = [
            number_of_sentiments
            for number_of_sentiments_list in department_number_of_sentiments
            for number_of_sentiments in number_of_sentiments_list]

        department_number_of_sentiments = sum(department_number_of_sentiments) if len(
            department_number_of_sentiments) > 0 else 0

        department_positive_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_positive_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_negative_sentiments_percentage = sum(
            [pd.read_csv(csv_department_file.csv_file_path)["department_negative_sentiments_percentage"].tolist()
             for csv_department_file in csv_department_files], [])

        department_positive_sentiments_percentage = round(
            sum(department_positive_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0
        department_negative_sentiments_percentage = round(
            sum(department_negative_sentiments_percentage) /
            len(actual_department_list), 2
        ) if len(actual_department_list) > 0 else 0

        # @desc: Get the total number of positive sentiments and negative sentiments based on the percentage
        department_positive_sentiments = round(
            department_number_of_sentiments * (department_positive_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_positive_sentiments_percentage else 0

        department_negative_sentiments = round(
            department_number_of_sentiments * (department_negative_sentiments_percentage / 100)) \
            if department_number_of_sentiments and department_negative_sentiments_percentage else 0

        school_year = school_year.replace("S.Y.", "")
        starting_year = school_year.split("-")[0]
        ending_year = school_year.split("-")[1]

        sentiment_details = [
            {"id": 1, "title": "Positive Sentiments",
             "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-smile-beam", "color50": "bg-green-50",
             "color500": "bg-green-500"},
            {"id": 2, "title": "Negative Sentiments",
             "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
             "year": starting_year + " - " + ending_year, "icon": "fas fa-face-frown", "color50": "bg-red-50",
             "color500": "bg-red-500"}
        ]

        return jsonify({"status": "success", "overall_sentiments": sentiment_details}), 200
