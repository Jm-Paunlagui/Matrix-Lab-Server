import base64
from io import BytesIO

import pandas as pd
import seaborn as sns
from flask import jsonify, session, Response
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from matrix.models.csv_file import CsvModel, CsvProfessorModel, CsvDepartmentModel
from matrix.models.user import User
from matrix.module import InputTextValidation


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

    # Add an option for school_year, school_semester, and csv_question for the dropdown
    school_year_dict.insert(0, {"id": 0, "school_year": "All"})
    school_semester_dict.insert(0, {"id": 0, "school_semester": "All"})
    csv_question_dict.insert(0, {"id": 0, "csv_question": "All"})

    return jsonify({
        "status": "success",
        "school_year": school_year_dict,
        "school_semester": school_semester_dict,
        "csv_question": csv_question_dict
    }), 200


def dashboard_data_csv():
    """@desc: Get the data of the csv files in the database."""
    # @desc: Get the Session to verify if the user is logged in.
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    if user_data.role != "admin":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    # @desc: Get the total number of csv files in the database
    csv_files = CsvModel.query.all()

    total_csv_files = len(csv_files)

    # @desc: Get the total number of csv files in the database that is being released
    total_released_csv_files = len(
        [csv_file for csv_file in csv_files if csv_file.flag_release == 1])
    total_unreleased_csv_files = len(
        [csv_file for csv_file in csv_files if csv_file.flag_release == 0])

    # @desc: Get the total number of csv files in the database that is being deleted temporarily
    total_deleted_csv_files = len(
        [csv_file for csv_file in csv_files if csv_file.flag_deleted == 1])

    data_csv = [
        {"id": 1, "title": "Total CSV Files",
            "value": total_csv_files, "icon": "fas fa-file-csv"},
        {"id": 2, "title": "Released CSV Files",
            "value": total_released_csv_files, "icon": "fas fa-file-csv"},
        {"id": 3, "title": "Unreleased CSV Files",
            "value": total_unreleased_csv_files, "icon": "fas fa-file-csv"},
        {"id": 4, "title": "Deleted CSV Files",
            "value": total_deleted_csv_files, "icon": "fas fa-file-csv"},
    ]

    return jsonify({
        "status": "success", "details": data_csv
    }), 200


def dashboard_data_professor():
    """@desc: Get the data of the professors in the database."""
    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=user_id).first()

    if user_data.role != "admin":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    # @desc: Read all the csv file in the database for professor

    # Count the number of users with the role of user
    total_users = User.query.filter_by(role="user").count()

    total_professors_active = User.query.filter_by(
        role="user", flag_active=1).count()
    total_professors_inactive = User.query.filter_by(
        role="user", flag_active=0).count()
    total_professors_locked = User.query.filter_by(
        role="user", flag_locked=1).count()
    total_professors_unlocked = User.query.filter_by(
        role="user", flag_locked=0).count()
    total_professors_deleted = User.query.filter_by(
        role="user", flag_deleted=1).count()
    total_professors_undeleted = User.query.filter_by(
        role="user", flag_deleted=0).count()
    # Count the number of users that has password in the database
    total_professors_without_password = User.query.filter_by(
        role="user", password=None).count()

    data_professor = [
        {"id": 1, "title": "Total Professors",
         "value": total_users, "icon": "fas fa-user-tie", "color": "bg-blue-500"},
        {"id": 2, "title": "Activated Professors",
         "value": total_professors_active, "icon": "fas fa-bolt", "color": "bg-teal-600"},
        {"id": 3, "title": "Unlocked Professors",
         "value": total_professors_unlocked, "icon": "fas fa-unlock", "color": "bg-teal-600"},
        {"id": 4, "title": "Undeleted Professors",
         "value": total_professors_undeleted, "icon": "fas fa-rotate", "color": "bg-teal-600"},
        {"id": 5, "title": "No Password",
         "value": total_professors_without_password, "icon": "fas fa-shield", "color": "bg-cyan-500"},
        {"id": 6, "title": "Deactivated Professors",
         "value": total_professors_inactive, "icon": "fas fa-circle-xmark", "color": "bg-red-600"},
        {"id": 7, "title": "Locked Professors",
         "value": total_professors_locked, "icon": "fas fa-lock", "color": "bg-red-600"},
        {"id": 8, "title": "Deleted Professors",
         "value": total_professors_deleted, "icon": "fas fa-trash", "color": "bg-red-600"},
    ]

    return jsonify({
        "status": "success", "details": data_professor
    }), 200


# @desc: CountVectorizer instance
vec = CountVectorizer()


def get_top_n_words(corpus, n=None):
    """
    @desc: Get the top n words in the corpus.

    Args:
        corpus (list): List of strings.
        n (int): Number of top words to return.

    Returns:
        list: List of top n words.
    """
    # @desc: Get the main text from the corpus

    if corpus is not None and len(corpus) > 0:
        main_text = [text[0] for text in corpus]

        # @desc: Get the sentiment from the corpus
        sentiment = [text[1] for text in corpus]

        vec.set_params(**{"ngram_range": (1, 1)})

        # @desc: Get the vectorized text
        bag_of_words = vec.fit_transform(main_text)

        # @desc: Get the sum of the vectorized text
        sum_words = bag_of_words.sum(axis=0)

        # @desc: Get the words from the vectorized text
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]

        # @desc: Sort the words from the vectorized text
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # @desc: Get the top n words
        top_n_words = words_freq[:n]

        # @desc: Get the top n words and the sentiment of the words and frequency of the words
        top_n_words_sentiment = [
            {"id": i + 1, "word": word, "sentiment": sentiment[i],
             "frequency": f"{str(frequency)} times"}
            for i, (word, frequency) in enumerate(top_n_words)]

        return top_n_words_sentiment
    return []


def get_top_n_bigrams(corpus, n=None):
    """
    @desc: Get the top n bigrams in the corpus.

    Args:
        corpus (list): List of strings.
        n (int): Number of top bigrams to return.

    Returns:
        list: List of top n bigrams.
    """
    if corpus is not None and len(corpus) > 0:
        # @desc: Get the main text from the corpus
        main_text = [text[0] for text in corpus]

        # @desc: Get the sentiment from the corpus
        sentiment = [text[1] for text in corpus]

        vec.set_params(**{"ngram_range": (2, 2)})

        # @desc: Get the vectorized text
        bag_of_words = vec.fit_transform(main_text)

        # @desc: Get the sum of the vectorized text
        sum_words = bag_of_words.sum(axis=0)

        # @desc: Get the words from the vectorized text
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]

        # @desc: Sort the words from the vectorized text
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # @desc: Get the top n words
        top_n_words = words_freq[:n]

        # @desc: Get the top n words and the sentiment of the words and frequency of the words
        top_n_words_sentiment = [
            {"id": i + 1, "word": word, "sentiment": sentiment[i],
             "frequency": f"{str(frequency)} times"}
            for i, (word, frequency) in enumerate(top_n_words)]

        return top_n_words_sentiment
    return []


def get_top_n_trigrams(corpus, n=None):
    """
    @desc: Get the top n trigrams in the corpus.

    Args:
        corpus (list): List of strings.
        n (int): Number of top trigrams to return.

    Returns:
        list: List of top n trigrams.
    """
    if corpus is not None and len(corpus) > 0:
        # @desc: Get the main text from the corpus
        main_text = [text[0] for text in corpus]

        # @desc: Get the sentiment from the corpus
        sentiment = [text[1] for text in corpus]

        vec.set_params(**{"ngram_range": (3, 3)})

        # @desc: Get the vectorized text
        bag_of_words = vec.fit_transform(main_text)

        # @desc: Get the sum of the vectorized text
        sum_words = bag_of_words.sum(axis=0)

        # @desc: Get the words from the vectorized text
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]

        # @desc: Sort the words from the vectorized text
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # @desc: Get the top n words
        top_n_words = words_freq[:n]

        # @desc: Get the top n words and the sentiment of the words and frequency of the words
        top_n_words_sentiment = [
            {"id": i + 1, "word": word, "sentiment": sentiment[i],
             "frequency": f"{str(frequency)} times"}
            for i, (word, frequency) in enumerate(top_n_words)]

        return top_n_words_sentiment
    return []


def computation(sentiment_converted_list=None, polarity_list=None, review_length_list=None, wordcloud_list=None,
                wordcloud_list_with_sentiment=None) -> tuple[str, str, str, list[tuple[str, str]]]:
    """
    @desc: Get the computation of the sentiment, polarity, review length, wordcloud, and wordcloud with sentiment.

    Args:
        sentiment_converted_list (list): List of sentiment converted.
        polarity_list (list): List of polarity.
        review_length_list (list): List of review length.
        wordcloud_list (list): List of wordcloud.
        wordcloud_list_with_sentiment (list): List of wordcloud with sentiment.

    Returns:
        tuple: Tuple of the computation of the sentiment, polarity, review length, wordcloud,
        and wordcloud with sentiment.
    """
    sentiment_converted_list = [sentiment for sentiment_list in sentiment_converted_list
                                for sentiment in sentiment_list]
    polarity_list = [
        polarity for polarity_list in polarity_list for polarity in polarity_list]
    review_length_list = [review_length for review_length_list in review_length_list
                          for review_length in review_length_list]
    wordcloud_list = [
        wordcloud for wordcloud_list in wordcloud_list for wordcloud in wordcloud_list]
    wordcloud_list_with_sentiment = [
        wordcloud for wordcloud_list in wordcloud_list_with_sentiment for wordcloud in wordcloud_list]

    # Plot the graph using matplotlib and seaborn library
    plt.figure(figsize=(8, 5))
    plt.title("Sentiment vs Polarity")
    plt.xlabel("Sentiment")
    plt.ylabel("Polarity")
    sns.set_style("whitegrid")
    if len(sentiment_converted_list) == 0 or len(polarity_list) == 0:
        sns.boxplot(x=[0], y=[0])
    else:
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
    if len(sentiment_converted_list) == 0 or len(review_length_list) == 0:
        sns.boxplot(x=[0], y=[0])
    else:
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

    # # @desc: Remove the float found in the wordcloud_list
    wordcloud_list = [str(wordcloud) for wordcloud in wordcloud_list]

    # @desc: Remove the float found in the wordcloud_list_with_sentiment and convert it to string format
    wordcloud_list_with_sentiment = [
        (str(wordcloud[0]), str(wordcloud[1])) for wordcloud in wordcloud_list_with_sentiment]

    # @desc: Get the overall word cloud 540 ?? 338 text color is only green and red and the background color is white
    if len(wordcloud_list) == 0:
        wordcloud = WordCloud(width=540, height=338, background_color="white",
                              colormap="tab10", collocations=False).generate("No Word Cloud")
    else:
        wordcloud = WordCloud(width=540, height=338, random_state=21, max_font_size=110, colormap="RdYlGn",
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

    return sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment


def get_starting_ending_year(csv_files: list) -> tuple[str, str]:
    """
    @desc: Get the starting and ending year of the csv files.

    Args:
        csv_files (list): List of csv files.

    Returns:
        tuple: Tuple of the starting and ending year of the csv files.
    """
    # @desc: Get the year of the csv file based on the list of csv files
    starting_year = csv_files[0].school_year.split(
        "-")[0] if len(csv_files) > 0 else "----"
    ending_year = csv_files[-1].school_year.split(
        "-")[1] if len(csv_files) > 0 else "----"
    # desc: remove the SY from the school year
    starting_year = starting_year.replace(
        "SY", "") if len(csv_files) > 0 else "----"
    ending_year = ending_year.replace(
        "SY", "") if len(csv_files) > 0 else "----"

    return starting_year, ending_year


def department_positive_and_negative_sentiment(csv_department_files: list) -> list:
    """
    @desc: Get the positive and negative sentiment of the csv files.

    Args:
        csv_department_files (list): List of csv files.

    Returns:
        list: List of the positive and negative sentiment of the csv files.
    """
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

    starting_year, ending_year = get_starting_ending_year(csv_department_files)

    sentiment_details = [
        {"id": 1, "title": "Positive Sentiments",
         "value": f"{department_positive_sentiments:,}", "percentage": department_positive_sentiments_percentage,
         "year": str(starting_year) + " - " + str(ending_year), "icon": "fas fa-face-smile-beam",
         "color50": "bg-green-50", "color500": "bg-green-500"},
        {"id": 2, "title": "Negative Sentiments",
         "value": f"{department_negative_sentiments:,}", "percentage": department_negative_sentiments_percentage,
         "year": str(starting_year) + " - " + str(ending_year), "icon": "fas fa-face-frown", "color50": "bg-red-50",
         "color500": "bg-red-500"}
    ]

    return sentiment_details


def professor_positive_and_negative_sentiment(csv_professor_files: list, evaluatee_name: str) -> list:
    """
    Get the positive and negative sentiments of the professor based on the csv files in the professor_analysis_csv_files
    directory, and it will be based on the evaluatee_name of the professor.

    Args:
        csv_professor_files: The list of the csv files.
        evaluatee_name: The name of the professor.

    Returns:
        list: The list of the positive and negative sentiments of the professor.
    """
    # # Get all the files first in the professor_analysis_csv_files directory

    # Based on the evaluatee_name, get the all its columns in the csv files
    professor_analysis_csv_files = [
        pd.read_csv(csv_professor_file.csv_file_path)[
            pd.read_csv(csv_professor_file.csv_file_path)["evaluatee_list"] == evaluatee_name]
        for csv_professor_file in csv_professor_files]

    # Computations Starts Here
    # @desc: Get the total number of evaluatee_overall_sentiments of the professor
    evaluatee_overall_sentiments = [
        professor_analysis_csv_file["evaluatee_overall_sentiment"].tolist()
        for professor_analysis_csv_file in professor_analysis_csv_files]

    evaluatee_overall_sentiments = [
        evaluatee_overall_sentiment
        for evaluatee_overall_sentiment_list in evaluatee_overall_sentiments

        for evaluatee_overall_sentiment in evaluatee_overall_sentiment_list]

    evaluatee_overall_sentiments = round(sum(
        evaluatee_overall_sentiments) / len(evaluatee_overall_sentiments), 2) \
        if len(evaluatee_overall_sentiments) > 0 else 0

    evaluatee_number_of_sentiments = [
        professor_analysis_csv_file["evaluatee_number_of_sentiments"].tolist()
        for professor_analysis_csv_file in professor_analysis_csv_files]

    evaluatee_number_of_sentiments = [
        evaluatee_number_of_sentiment
        for evaluatee_number_of_sentiment_list in evaluatee_number_of_sentiments
        for evaluatee_number_of_sentiment in evaluatee_number_of_sentiment_list]

    evaluatee_number_of_sentiments = sum(evaluatee_number_of_sentiments) if len(
        evaluatee_number_of_sentiments) > 0 else 0

    evaluatee_positive_sentiments_percentage = [
        professor_analysis_csv_file["evaluatee_positive_sentiments_percentage"].tolist(
        )
        for professor_analysis_csv_file in professor_analysis_csv_files]

    evaluatee_positive_sentiments_percentage = [
        evaluatee_positive_sentiment_percentage
        for evaluatee_positive_sentiment_percentage_list in evaluatee_positive_sentiments_percentage
        for evaluatee_positive_sentiment_percentage in evaluatee_positive_sentiment_percentage_list]

    evaluatee_positive_sentiments_percentage = round(
        sum(evaluatee_positive_sentiments_percentage) / len(evaluatee_positive_sentiments_percentage), 2) \
        if len(evaluatee_positive_sentiments_percentage) > 0 else 0

    evaluatee_negative_sentiments_percentage = [
        professor_analysis_csv_file["evaluatee_negative_sentiments_percentage"].tolist(
        )
        for professor_analysis_csv_file in professor_analysis_csv_files]

    evaluatee_negative_sentiments_percentage = [
        evaluatee_negative_sentiment_percentage
        for evaluatee_negative_sentiment_percentage_list in evaluatee_negative_sentiments_percentage
        for evaluatee_negative_sentiment_percentage in evaluatee_negative_sentiment_percentage_list]

    evaluatee_negative_sentiments_percentage = round(
        sum(evaluatee_negative_sentiments_percentage) / len(evaluatee_negative_sentiments_percentage), 2) \
        if len(evaluatee_negative_sentiments_percentage) > 0 else 0

    starting_year, ending_year = get_starting_ending_year(csv_professor_files)

    sentiment_details = [
        {"id": 1, "title": "Positive Sentiments",
         "value": f"{evaluatee_positive_sentiments_percentage:,}",
         "percentage": evaluatee_positive_sentiments_percentage,
         "year": str(starting_year) + " - " + str(ending_year), "icon": "fas fa-face-smile-beam",
         "color50": "bg-green-50", "color500": "bg-green-500"},
        {"id": 2, "title": "Negative Sentiments",
         "value": f"{evaluatee_negative_sentiments_percentage:,}",
         "percentage": evaluatee_negative_sentiments_percentage,
         "year": str(starting_year) + " - " + str(ending_year), "icon": "fas fa-face-frown", "color50": "bg-red-50",
         "color500": "bg-red-500"}
    ]

    return sentiment_details


def analysis_admin(csv_files: tuple) -> tuple[str, str, str, list[tuple[str, str]]]:
    """
    Get the analysis of the admin.

    Args:
        csv_files (tuple): The tuple of the csv files.

    Returns:
        tuple[str, str, str, list[tuple[str, str]]]: The tuple of the admins' analysis.
    """
    csv_file_path = [csv_file.csv_file_path for csv_file in csv_files]

    # @dec: Read all the csv files in the database
    csv_files = [pd.read_csv(csv_file) for csv_file in csv_file_path]

    sentiment_converted_list = []
    polarity_list = []
    review_length_list = []
    wordcloud_list = []
    wordcloud_list_with_sentiment = []

    for csv_file in csv_files:
        sentiment_converted_list.append(
            csv_file["sentiment_converted"].tolist())
        polarity_list.append(csv_file["polarity"].tolist())
        review_length_list.append(csv_file["review_len"].tolist())
        wordcloud_list.append(
            csv_file["sentence_remove_stopwords"].tolist())
        # Combine word list and sentiment_converted list into a list of tuples
        wordcloud_list_with_sentiment.append(
            list(zip(csv_file["sentence_remove_stopwords"].tolist(), csv_file["sentiment_converted"].tolist())))

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment = \
        computation(sentiment_converted_list=sentiment_converted_list, polarity_list=polarity_list,
                    review_length_list=review_length_list, wordcloud_list=wordcloud_list,
                    wordcloud_list_with_sentiment=wordcloud_list_with_sentiment)

    return sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment


def analysis_user(csv_files: tuple, evaluatee_name: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    """@desc: Get the analysis of the user using matplotlib and seaborn library

    Args:
        csv_files (tuple): Tuple of csv files
        evaluatee_name (str): Name of the user

    Returns:
        tuple[str, str, str, list[tuple[str, str]]]: The tuple of the users' analysis.
    """
    csv_file_path = [csv_file.csv_file_path for csv_file in csv_files]

    # @dec: Read all the csv files in the database
    csv_files = [pd.read_csv(csv_file) for csv_file in csv_file_path]

    # @desc: Get the Sentiment vs Polarity of the csv files using matplotlib and seaborn library
    sentiment_converted_list = []
    polarity_list = []
    review_length_list = []
    wordcloud_list = []
    wordcloud_list_with_sentiment = []

    # Based on the evaluatee_name, get the all its columns in the csv files
    for csv_file in csv_files:
        sentiment_converted_list.append(
            csv_file[csv_file["evaluatee"] == evaluatee_name]["sentiment_converted"].tolist())
        polarity_list.append(
            csv_file[csv_file["evaluatee"] == evaluatee_name]["polarity"].tolist())
        review_length_list.append(
            csv_file[csv_file["evaluatee"] == evaluatee_name]["review_len"].tolist())
        wordcloud_list.append(
            csv_file[csv_file["evaluatee"] == evaluatee_name]["sentence_remove_stopwords"].tolist())
        wordcloud_list_with_sentiment.append(
            list(zip(csv_file[csv_file["evaluatee"] == evaluatee_name]["sentence_remove_stopwords"].tolist(),
                     csv_file[csv_file["evaluatee"] == evaluatee_name]["sentiment_converted"].tolist())))

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment = \
        computation(sentiment_converted_list=sentiment_converted_list, polarity_list=polarity_list,
                    review_length_list=review_length_list, wordcloud_list=wordcloud_list,
                    wordcloud_list_with_sentiment=wordcloud_list_with_sentiment)

    return sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment


def analysis_options_admin(school_year: str, school_semester: str, csv_question: str) \
        -> tuple[Response, int]:
    """
    Get the analysis of the admin using matplotlib and seaborn library.

    Args:
        school_year (str): The school year.
        school_semester (str): The school semester.
        csv_question (str): The csv question.

    Returns:
        tuple[Response, int]: The response and the status code and the analysis of the csv files
    """
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    if school_year == "All" and school_semester == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and school_semester == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(
            csv_question=csv_question).all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(csv_question=csv_question).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(
            school_semester=school_semester).all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            school_semester=school_semester).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(
            school_year=school_year).all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(school_year=school_year).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(csv_question=csv_question,
                                                                  school_semester=school_semester).all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            csv_question=csv_question, school_semester=school_semester).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(csv_question=csv_question,
                                                                  school_year=school_year).all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            csv_question=csv_question, school_year=school_year).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_department_files = CsvDepartmentModel.query.filter_by(school_year=school_year,
                                                                  school_semester=school_semester).all()

        sentiment_details = department_positive_and_negative_sentiment(
            csv_department_files)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            school_year=school_year, school_semester=school_semester).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = analysis_admin(csv_files)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    # @desc: Read all the csv file in the database for department
    csv_department_files = CsvDepartmentModel.query.filter_by(csv_question=csv_question,
                                                              school_semester=school_semester,
                                                              school_year=school_year).all()

    sentiment_details = department_positive_and_negative_sentiment(
        csv_department_files)

    # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
    csv_files = CsvModel.query.filter_by(csv_question=csv_question, school_semester=school_semester,
                                         school_year=school_year).all()

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
        wordcloud_list_with_sentiment = analysis_admin(csv_files)

    return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                    "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                    "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                    "image_path_wordcloud": wordcloud_encoded,
                    "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                    "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                    "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                    }), 200


def analysis_options_user(school_year: str, school_semester: str, csv_question: str) -> tuple[Response, int]:
    """
    @desc: This function is used to get the analysis of the csv files for the user

    Args:
        school_year (str): The school year of the csv files
        school_semester (str): The school semester of the csv files
        csv_question (str): The csv question of the csv files

    Returns:
        tuple[Response, int]: The response and the status code and the analysis of the csv files
    """
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    user_id: int = session.get('user_id')

    if user_id is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    user_data: User = User.query.with_entities(
        User.role, User.full_name).filter_by(user_id=user_id).first()

    if user_data.role != "user":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    converted_full_name = InputTextValidation(
        user_data.full_name).to_csv_professor_name()

    if school_year == "All" and school_semester == "All" and csv_question == "All":
        csv_files = CsvProfessorModel.query.all()
        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        csv_files = CsvModel.query.all()
        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and school_semester == "All":
        # @desc: Read all the csv file in the database for department
        csv_files = CsvProfessorModel.query.filter_by(
            csv_question=csv_question).all()

        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(csv_question=csv_question).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_files = CsvProfessorModel.query.filter_by(
            school_semester=school_semester).all()

        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            school_semester=school_semester).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All" and csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_files = CsvProfessorModel.query.filter_by(
            school_year=school_year).all()

        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(school_year=school_year).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All":
        # @desc: Read all the csv file in the database for department
        csv_files = CsvProfessorModel.query.filter_by(
            csv_question=csv_question, school_semester=school_semester).all()

        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            csv_question=csv_question, school_semester=school_semester).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All":
        # @desc: Read all the csv file in the database for department
        csv_files = CsvProfessorModel.query.filter_by(
            csv_question=csv_question, school_year=school_year).all()

        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            csv_question=csv_question, school_year=school_year).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if csv_question == "All":
        # @desc: Read all the csv file in the database for department
        csv_files = CsvProfessorModel.query.filter_by(
            school_year=school_year, school_semester=school_semester).all()

        sentiment_details = professor_positive_and_negative_sentiment(
            csv_files, converted_full_name)

        # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
        csv_files = CsvModel.query.filter_by(
            school_year=school_year, school_semester=school_semester).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
            = analysis_user(csv_files, converted_full_name)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    # @desc: Read all the csv file in the database for department
    csv_files = CsvProfessorModel.query.filter_by(csv_question=csv_question, school_semester=school_semester,
                                                  school_year=school_year).all()

    sentiment_details = professor_positive_and_negative_sentiment(
        csv_files, converted_full_name)

    # @desc: Get Sentiment vs Polarity of the csv files using matplotlib and seaborn library
    csv_files = CsvModel.query.filter_by(csv_question=csv_question, school_semester=school_semester,
                                         school_year=school_year).all()

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment \
        = analysis_user(csv_files, converted_full_name)

    return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                    "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                    "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                    "image_path_wordcloud": wordcloud_encoded,
                    "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                    "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                    "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                    }), 200
