import base64
from io import BytesIO

import seaborn as sns
from flask import Response, jsonify
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from extensions import db
from matrix.models.csv_file import (CsvAnalyzedSentiment,
                                    CsvDepartmentSentiment, CsvModelDetail,
                                    CsvProfessorSentiment)
from matrix.models.user import User
from matrix.module import InputTextValidation, verify_authenticated_token


def options_read_single_data_dashboard():
    """Options for the read single data route."""
    csv_file = CsvModelDetail.query.filter(
        CsvModelDetail.flag_deleted == False
    ).all()

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


def dashboard_data_csv(token: str):
    """@desc: Get the data of the csv files in the database."""
    # @desc: Get the Session to verify if the user is logged in.

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=verified_token["id"]).first()

    if user_data.role != "admin":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    # @desc: Get the total number of csv files in the database
    csv_files = CsvModelDetail.query.filter(
        CsvModelDetail.flag_deleted == False
    ).all()

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


def dashboard_data_professor(token: str):
    """@desc: Get the data of the professors in the database."""

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role).filter_by(user_id=verified_token["id"]).first()

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
                wordcloud_list_with_sentiment=None, title=None) -> tuple[str, str, str, list[tuple[str, str]]]:
    """
    @desc: Get the computation of the sentiment, polarity, review length, wordcloud, and wordcloud with sentiment.

    Args:
        sentiment_converted_list (list): List of sentiment converted.
        polarity_list (list): List of polarity.
        review_length_list (list): List of review length.
        wordcloud_list (list): List of wordcloud.
        wordcloud_list_with_sentiment (list): List of wordcloud with sentiment.
        title (str): Title of the computation.

    Returns:
        tuple: Tuple of the computation of the sentiment, polarity, review length, wordcloud,
        and wordcloud with sentiment.
    """

    # Plot the graph using matplotlib and seaborn library
    plt.figure(figsize=(8, 5))
    plt.title(f"Sentiment vs Polarity ({title})")
    plt.xlabel("Sentiment")
    plt.ylabel("Polarity")
    sns.set_style("whitegrid")
    sns.set_palette(['#ef4444', '#22c55e'])
    if len(sentiment_converted_list) == 0 or len(polarity_list) == 0:
        sns.boxplot(x=[0], y=[0])
    else:
        ax = sns.boxplot(
            x=sentiment_converted_list,
            y=polarity_list,
            showfliers=True, showmeans=True,
            flierprops=dict(marker='d', markerfacecolor='black',
                            markersize=5, linestyle='none'),
            medianprops=dict(color='yellow'),
            meanprops=dict(
                marker='o', markerfacecolor='#06b6d4', markersize=5),
            capprops=dict(color='black'),
        )

        # Legend for the ylabels of the boxplot graph
        xticks = [0, 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels(["Negative", "Positive"])
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter(["Negative", "Positive"]))

        # Add the legend for the boxplot components
        flier_elements = [
            Line2D([0], [0], marker='d', color='w',
                   markerfacecolor='black', markersize=5, label='Outliers'),
        ]
        median_elements = [
            Line2D([0], [0], color='yellow', label='Median')]
        min_max_elements = [
            Line2D([0], [0], color='black', label='Min/Max')]
        quartile_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#ef4444', label='25%-75% Negative'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#22c55e', label='25%-75% Positive')]

        mean_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#06b6d4', label='Mean')]

        ax.legend(handles=flier_elements + median_elements + min_max_elements + quartile_elements + mean_elements,
                  title="Legend", loc="best")

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
    plt.title(f"Sentiment vs Review Length ({title})")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Length")
    sns.set_style("whitegrid")
    sns.set_palette(['#ef4444', '#22c55e'])
    if len(sentiment_converted_list) == 0 or len(review_length_list) == 0:
        sns.boxplot(x=[0], y=[0])
    else:
        ax = sns.pointplot(x=sentiment_converted_list, y=review_length_list,
                           hue=sentiment_converted_list, dodge=False,)

        xticks = [0, 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels(["Negative", "Positive"])
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter(["Negative", "Positive"]))

        elements = [
            Line2D([0, 0], [0, 1], marker='o', color='w',
                   markerfacecolor='#ef4444', label='Negative'),
            Line2D([0, 0], [0, 1], marker='o', color='w',
                   markerfacecolor='#22c55e', label='Positive'),
        ]

        ax.legend(handles=elements, title="Legend",
                  loc="best", bbox_to_anchor=(1, 1))

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

    # @desc: Remove the float found in the wordcloud_list_with_sentiment and convert it to string format
    wordcloud_list_with_sentiment = [
        (str(wordcloud[0]), str(wordcloud[1])) for wordcloud in wordcloud_list_with_sentiment]

    # @desc: Get the overall word cloud 540 × 338 text color is only green and red and the background color is white
    if len(wordcloud_list) == 0:
        wordcloud = WordCloud(width=540, height=338, background_color="white",
                              colormap="tab10", collocations=False
                              ).generate("No Word Cloud")
    else:
        wordcloud = WordCloud(width=540, height=338, random_state=21, max_font_size=110, colormap="RdYlGn",
                              background_color="white").generate(" ".join(wordcloud_list))

    # Save the figure to a BytesIO object
    buf_wordcloud = BytesIO()
    plt.figure(figsize=(8, 5))
    plt.title(f"Word Cloud ({title})")
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


def pancore_compute(sentiments: list[tuple[int, int, str, int, float, float]],
                    starting_year: str, ending_year: str, evaluatee_name: str | None):
    """
    Calculate the total number of sentiments.

    Args:
        sentiments (list[tuple[int, int, str, int, float, float]]): List of sentiments.
        starting_year (int): The starting year.
        ending_year (int): The ending year.
        evaluatee_name (str): The name of the evaluatee.

    Returns:
        int: The total number of sentiments.
        int: The total number of positive sentiments.
        int: The total number of negative sentiments.
        float: The percentage of positive sentiments.
        float: The percentage of negative sentiments.
        evaluatee_name (str): The name of the evaluatee.
    """
    total_sentiments = 0
    total_positive_sentiments = 0
    total_negative_sentiments = 0
    positive_sentiment = 0
    negative_sentiment = 0
    if evaluatee_name is not None:
        # Depending on the evaluatee name, get the total number of sentiments
        total_distinct_ids = len(
            {sentiment[0] for sentiment in sentiments if sentiment[2] == evaluatee_name})

        total_sentiments = sum(
            [sentiment[3] for sentiment in sentiments if sentiment[2] == evaluatee_name])

        positive_sentiment = round(
            sum([sentiment[4] for sentiment in sentiments if sentiment[2]
                 == evaluatee_name]) / total_distinct_ids,
            2) if total_distinct_ids != 0 else 0

        negative_sentiment = round(
            sum([sentiment[5] for sentiment in sentiments if sentiment[2]
                 == evaluatee_name]) / total_distinct_ids,
            2) if total_distinct_ids != 0 else 0

        total_positive_sentiments = round(
            total_sentiments * positive_sentiment / 100, 2)

        total_negative_sentiments = round(
            total_sentiments * negative_sentiment / 100, 2)

    if evaluatee_name is None:
        total_distinct_ids = len(sentiments)

        total_sentiments = sum(
            [sentiment[3] for sentiment in sentiments])

        positive_sentiment = round(
            sum([sentiment[4] for sentiment in sentiments]) / total_distinct_ids, 2) if total_distinct_ids != 0 else 0

        negative_sentiment = round(
            sum([sentiment[5] for sentiment in sentiments]) / total_distinct_ids, 2) if total_distinct_ids != 0 else 0

        total_positive_sentiments = round(
            total_sentiments * positive_sentiment / 100, 2)

        total_negative_sentiments = round(
            total_sentiments * negative_sentiment / 100, 2)

    sentiment_details = [
        {
            "id": 1, "title": "Positive Sentiments",
            "value": f"{total_positive_sentiments:,} / {total_sentiments:,}",
            "percentage": positive_sentiment,
            "year": str(starting_year) + " - " + str(ending_year), "icon": "fas fa-face-smile-beam",
            "color50": "bg-green-50", "color500": "bg-green-500"
        },
        {
            "id": 2, "title": "Negative Sentiments",
            "value": f"{total_negative_sentiments:,} / {total_sentiments:,}",
            "percentage": negative_sentiment,
            "year": str(starting_year) + " - " + str(ending_year), "icon": "fas fa-face-frown",
            "color50": "bg-red-50", "color500": "bg-red-500"
        }
    ]

    # Top departments
    number_of_sentiments, positive_sentiments_percentage, negative_sentiments_percentage = [], [], []
    no_of_evaluated = db.session.query(CsvModelDetail).filter(CsvModelDetail.flag_deleted == False).count()
    # Distinct departments
    names = list({sentiment[2] for sentiment in sentiments})

    for name in names:
        number_of_sentiments.append(
            sum([sentiment[3]
                 for sentiment in sentiments if sentiment[2] == name])
        )
        # Recalculate the percentage of positive and negative sentiments by department and divide by the number of
        # evaluated files
        positive_sentiments_percentage.append(
            round(
                sum([sentiment[4] for sentiment in sentiments if sentiment[2] == name]) / no_of_evaluated, 2)
        ) if no_of_evaluated > 0 else 0
        negative_sentiments_percentage.append(
            round(
                sum([sentiment[5] for sentiment in sentiments if sentiment[2] == name]) / no_of_evaluated, 2)
        ) if no_of_evaluated > 0 else 0

    department_details = [
        {
            "id": index,
            "name": name,
            "positive_sentiments_percentage":
                f"{positive_sentiments_percentage[names.index(name)]}%",
            "negative_sentiments_percentage":
                f"{negative_sentiments_percentage[names.index(name)]}%",
            "number_of_sentiments": f"{number_of_sentiments[names.index(name)]:,} / {total_sentiments:,}",
        } for index, name in enumerate(
            sorted(names, key=lambda x: positive_sentiments_percentage[names.index(x)],
                   reverse=True), start=0)] if positive_sentiments_percentage else []

    if evaluatee_name is not None:
        return sentiment_details
    return sentiment_details, department_details


def remove_none_values(sentiment_converted_list: list, polarity_list: list, review_length_list: list,
                       wordcloud_list: list, wordcloud_list_with_sentiment: list) -> tuple:
    """
    Remove the None values in the sentiment_converted_list, polarity_list, review_length_list, wordcloud_list,
    wordcloud_list_with_sentiment.

    Args:
        sentiment_converted_list: The list of the sentiment converted.
        polarity_list: The list of the polarity.
        review_length_list: The list of the review length.
        wordcloud_list: The list of the wordcloud.
        wordcloud_list_with_sentiment: The list of the wordcloud with sentiment.

    Returns:
        tuple: The tuple of the sentiment_converted_list, polarity_list, review_length_list, wordcloud_list,
        wordcloud_list_with_sentiment.
    """
    sentiment_converted_list = [
        sentiment_converted for sentiment_converted in sentiment_converted_list if sentiment_converted is not None]

    polarity_list = [
        polarity for polarity in polarity_list if polarity is not None]

    review_length_list = [
        review_length for review_length in review_length_list if review_length is not None]

    wordcloud_list = [
        wordcloud for wordcloud in wordcloud_list if wordcloud is not None]

    wordcloud_list_with_sentiment = [
        wordcloud_with_sentiment for wordcloud_with_sentiment in wordcloud_list_with_sentiment
        if wordcloud_with_sentiment is not None]

    return sentiment_converted_list, polarity_list, review_length_list, wordcloud_list, wordcloud_list_with_sentiment


def core_analysis(
        analysis: list[tuple[int, int, int, float, str, int]] | list[tuple[int, int, str, int, float, str, int]],
        evaluatee_name: str | None, title=None) -> tuple[str, str, str, list[tuple[str, str]]]:
    """
    Generate the analysis of the sentiments.

    Args:
        analysis (list[tuple[int, int, int, float, str, int]]): The list of the analysis.
        evaluatee_name (str): The name of the professor.
        title (str): additional

    Returns:
        list[dict]: The list of the analysis.
    """
    # Combine all the analysis into one list each
    sentiment_converted_list = []
    polarity_list = []
    review_length_list = []
    wordcloud_list = []
    wordcloud_list_with_sentiment = []

    if evaluatee_name is not None:
        for sentiment in analysis:
            sentiment_converted_list.append(
                sentiment[3] if sentiment[2] == evaluatee_name else None)
            polarity_list.append(
                sentiment[4] if sentiment[2] == evaluatee_name else None)
            review_length_list.append(
                sentiment[6] if sentiment[2] == evaluatee_name else None)
            wordcloud_list.append(
                sentiment[5] if sentiment[2] == evaluatee_name else None)
            wordcloud_list_with_sentiment.append((sentiment[5], sentiment[3])
                                                 if sentiment[2] == evaluatee_name else None)
    else:
        for sentiment in analysis:
            sentiment_converted_list.append(sentiment[2])
            polarity_list.append(sentiment[3])
            review_length_list.append(sentiment[5])
            wordcloud_list.append(sentiment[4])
            wordcloud_list_with_sentiment.append((sentiment[4], sentiment[2]))
    sentiment_converted_list, polarity_list, review_length_list, wordcloud_list, wordcloud_list_with_sentiment = \
        remove_none_values(
            sentiment_converted_list, polarity_list, review_length_list, wordcloud_list,
            wordcloud_list_with_sentiment)

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment = \
        computation(sentiment_converted_list=sentiment_converted_list, polarity_list=polarity_list,
                    review_length_list=review_length_list, wordcloud_list=wordcloud_list,
                    wordcloud_list_with_sentiment=wordcloud_list_with_sentiment, title=title)

    return sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, wordcloud_list_with_sentiment


def analysis_options_admin(school_year: str, school_semester: str, csv_question: str, token:str) \
        -> tuple[Response, int]:
    """
    Get the analysis of the admin using matplotlib and seaborn library.

    Args:
        school_year (str): The school year.
        school_semester (str): The school semester.
        csv_question (str): The csv question.
        token (str): The token of the user.

    Returns:
        tuple[Response, int]: The response and the status code and the analysis of the csv files
    """
    school_yearr = school_year
    school_semesterr = school_semester
    csv_questionn = csv_question
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.full_name, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    if user_data.role != "admin":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    if user_data.verified_email != "Verified":
        return jsonify({"status": "error", "message": "You are not verified to access this page."}), 401

    if school_year == "All" and school_semester == "All" and csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())

        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id ==
                 CsvDepartmentSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id ==
                 CsvAnalyzedSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title="Overall")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and school_semester == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())

        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title="Overall " + csv_questionn)
        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
            filter(CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title=f"{school_semesterr} for Overall")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All" and csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title=f"{school_yearr} for Overall")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title=f"{school_semesterr} for {csv_questionn}")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title=f"{school_yearr} for {csv_questionn} ")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
            CsvDepartmentSentiment.department_number_of_sentiments,
            CsvDepartmentSentiment.department_positive_sentiments_percentage,
            CsvDepartmentSentiment.department_negative_sentiments_percentage). \
            join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details, department_details = pancore_compute(
            sentiments, starting_year, ending_year, None)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
            CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
            CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, None, title=f"{school_year} - {school_semesterr} for Overall Topics")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "overall_departments": department_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    starting_year, ending_year = get_starting_ending_year(
        db.session.query(CsvModelDetail.school_year).filter(
            CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all())
    sentiments = db.session.query(
        CsvModelDetail.csv_id, CsvDepartmentSentiment.csv_id, CsvDepartmentSentiment.department,
        CsvDepartmentSentiment.department_number_of_sentiments,
        CsvDepartmentSentiment.department_positive_sentiments_percentage,
        CsvDepartmentSentiment.department_negative_sentiments_percentage). \
        join(CsvDepartmentSentiment, CsvModelDetail.csv_id == CsvDepartmentSentiment.csv_id). \
        filter(CsvModelDetail.school_year == school_year, CsvModelDetail.school_semester == school_semester,
               CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
    ).all()

    sentiment_details, department_details = pancore_compute(
        sentiments, starting_year, ending_year, None)

    analysis = db.session.query(
        CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.sentiment_converted,
        CsvAnalyzedSentiment.polarity, CsvAnalyzedSentiment.sentence_remove_stopwords,
        CsvAnalyzedSentiment.review_len). \
        join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
        filter(CsvModelDetail.csv_question == csv_question, CsvModelDetail.school_semester == school_semester,
               CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
    ).all()

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
        wordcloud_list_with_sentiment = core_analysis(
            analysis, None, title=f"{school_yearr} - {school_semesterr} for {csv_questionn}")

    return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                    "overall_departments": department_details,
                    "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                    "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                    "image_path_wordcloud": wordcloud_encoded,
                    "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                    "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                    "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                    }), 200


def analysis_options_user(school_year: str, school_semester: str, csv_question: str, token:str) -> tuple[Response, int]:
    """
    @desc: This function is used to get the analysis of the csv files for the user

    Args:
        school_year (str): The school year of the csv files
        school_semester (str): The school semester of the csv files
        csv_question (str): The csv question of the csv files
        token (str): The token of the user

    Returns:
        tuple[Response, int]: The response and the status code and the analysis of the csv files
    """
    school_yearr = school_year
    school_semesterr = school_semester
    csv_questionn = csv_question
    school_year = InputTextValidation(school_year).to_query_school_year()
    school_semester = InputTextValidation(
        school_semester).to_query_space_under()
    csv_question = InputTextValidation(csv_question).to_query_csv_question()

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    user_data: User = User.query.with_entities(
        User.role, User.full_name, User.verified_email).filter_by(user_id=verified_token["id"]).first()

    if user_data.role != "user":
        return jsonify({"status": "error", "message": "You are not authorized to access this page."}), 401

    converted_full_name = InputTextValidation(
        user_data.full_name).to_csv_professor_name()

    if user_data.verified_email != "Verified":
        return jsonify({"status": "error", "message": "You are not verified to access this page."}), 401

    if school_year == "All" and school_semester == "All" and csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())

        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id ==
                 CsvProfessorSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id ==
                 CsvAnalyzedSentiment.csv_id).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title="Overall")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and school_semester == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())

        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title="Overall " + csv_questionn)

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All" and csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
            filter(CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title=f"{school_semesterr} for Overall")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All" and csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title=f"{school_yearr} for Overall")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_year == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title=f"{school_semesterr} for {csv_questionn}")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if school_semester == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.csv_question == csv_question,
                   CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title=f"{school_yearr} for {csv_questionn} ")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    if csv_question == "All":
        starting_year, ending_year = get_starting_ending_year(
            db.session.query(CsvModelDetail.school_year).filter(
                CsvModelDetail.school_year == school_year).filter(
                CsvModelDetail.flag_deleted == False
            ).all())
        sentiments = db.session.query(
            CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
            CsvProfessorSentiment.evaluatee_number_of_sentiments,
            CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
            CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
            join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_details = pancore_compute(
            sentiments, starting_year, ending_year, converted_full_name)

        analysis = db.session.query(
            CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
            CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
            CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
            join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
            filter(CsvModelDetail.school_year == school_year,
                   CsvModelDetail.school_semester == school_semester).filter(
            CsvModelDetail.flag_deleted == False
        ).all()

        sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
            wordcloud_list_with_sentiment = core_analysis(
                analysis, converted_full_name, title=f"{school_yearr} - {school_semesterr} for Overall Topics")

        return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                        "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                        "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                        "image_path_wordcloud": wordcloud_encoded,
                        "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                        "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                        "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                        }), 200
    starting_year, ending_year = get_starting_ending_year(
        db.session.query(CsvModelDetail.school_year).filter(
            CsvModelDetail.school_year == school_year).filter(
            CsvModelDetail.flag_deleted == False
        ).all())
    sentiments = db.session.query(
        CsvModelDetail.csv_id, CsvProfessorSentiment.csv_id, CsvProfessorSentiment.professor,
        CsvProfessorSentiment.evaluatee_number_of_sentiments,
        CsvProfessorSentiment.evaluatee_positive_sentiments_percentage,
        CsvProfessorSentiment.evaluatee_negative_sentiments_percentage). \
        join(CsvProfessorSentiment, CsvModelDetail.csv_id == CsvProfessorSentiment.csv_id). \
        filter(CsvModelDetail.school_year == school_year, CsvModelDetail.school_semester == school_semester,
               CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
    ).all()

    sentiment_details = pancore_compute(
        sentiments, starting_year, ending_year, converted_full_name)

    analysis = db.session.query(
        CsvModelDetail.csv_id, CsvAnalyzedSentiment.csv_id, CsvAnalyzedSentiment.evaluatee,
        CsvAnalyzedSentiment.sentiment_converted, CsvAnalyzedSentiment.polarity,
        CsvAnalyzedSentiment.sentence_remove_stopwords, CsvAnalyzedSentiment.review_len). \
        join(CsvAnalyzedSentiment, CsvModelDetail.csv_id == CsvAnalyzedSentiment.csv_id). \
        filter(CsvModelDetail.csv_question == csv_question, CsvModelDetail.school_semester == school_semester,
               CsvModelDetail.csv_question == csv_question).filter(
            CsvModelDetail.flag_deleted == False
    ).all()

    sentiment_polarity_encoded, sentiment_review_length_encoded, wordcloud_encoded, \
        wordcloud_list_with_sentiment = core_analysis(
            analysis, converted_full_name, title=f"{school_yearr} - {school_semesterr} for {csv_questionn}")

    return jsonify({"status": "success", "overall_sentiments": sentiment_details,
                    "image_path_polarity_v_sentiment": sentiment_polarity_encoded,
                    "image_path_review_length_v_sentiment": sentiment_review_length_encoded,
                    "image_path_wordcloud": wordcloud_encoded,
                    "common_word": get_top_n_words(wordcloud_list_with_sentiment, 30),
                    "common_words": get_top_n_bigrams(wordcloud_list_with_sentiment, 30),
                    "common_phrase": get_top_n_trigrams(wordcloud_list_with_sentiment, 30),
                    }), 200
