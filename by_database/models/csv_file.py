from extensions import db


class CsvModelDetail(db.Model):
    """
    Csv model class attributes
    csv_id: Csv id number (primary key) (auto increment) bigint
    csv_question: Csv question varchar(255)
    csv_file_path: Csv file path text
    school_year: School year varchar(255)
    school_semester: School semester varchar(255)
    flag_deleted: Flag deleted boolean
    flag_release: Flag release boolean
    """

    __tablename__ = 'csv_model_detail_test'
    csv_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    csv_question = db.Column(db.String(255))
    school_year = db.Column(db.String(255))
    school_semester = db.Column(db.String(255))
    flag_deleted = db.Column(db.Boolean, default=False)
    flag_release = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"CsvModel(csv_id={self.csv_id}, csv_question={self.csv_question}, school_year={self.school_year}, " \
               f"school_semester={self.school_semester}, flag_deleted={self.flag_deleted}, " \
               f"flag_release={self.flag_release})"


class CsvAnalyzedSentiment(db.Model):
    """
    Csv analyzed sentiment model class attributes
    csv_analyzed_sentiment_id: Csv analyzed sentiment id number (primary key) (auto increment) bigint
    csv_id: Csv id number (foreign key) (not null) bigint relationship with csv model
    evaluatee: Evaluatee varchar(255)
    email: Email varchar(255) (unique)
    department: Department varchar(255)
    course_code: Course code varchar(255)
    sentence: Sentence text (not null)
    sentiment: Sentiment varchar(255)
    sentiment_converted: Sentiment integer 1 or 0
    sentence_remove_stopwords: Sentence remove stopwords text
    review_len: Review length integer
    word_count: Word count integer
    polarity: Polarity float
    """

    __tablename__ = 'csv_analyzed_sentiment_test'
    csv_analyzed_sentiment_id = db.Column(
        db.BigInteger, primary_key=True, autoincrement=True)
    csv_id = db.Column(db.BigInteger, nullable=False)
    evaluatee = db.Column(db.String(255))
    department = db.Column(db.String(255))
    course_code = db.Column(db.String(255))
    sentence = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.Float)
    sentiment_converted = db.Column(db.Integer)
    sentence_remove_stopwords = db.Column(db.Text)
    review_len = db.Column(db.Integer)
    word_count = db.Column(db.Integer)
    polarity = db.Column(db.Float)

    def __repr__(self):
        return f"CsvAnalyzedSentiment(csv_analyzed_sentiment_id={self.csv_analyzed_sentiment_id}, " \
               f"csv_id={self.csv_id}, evaluatee={self.evaluatee}, department={self.department}, " \
               f"course_code={self.course_code}, sentence={self.sentence}, sentiment={self.sentiment}, " \
               f"sentiment_converted={self.sentiment_converted}, " \
               f"sentence_remove_stopwords={self.sentence_remove_stopwords}, review_len={self.review_len}, " \
               f"word_count={self.word_count}, polarity={self.polarity})"

# class CsvProfessorSentiment(db.Model):
#     """
#     Csv professor sentiment model class attributes
#     csv_professor_sentiment_id: Csv professor sentiment id number (primary key) (auto increment) bigint
#     csv_id: Csv id number bigint
#     professor: Professor varchar(255) evaluatee
#     evaluatee_overall_sentiment: Evaluatee overall sentiment float
#     evaluatee_department: Evaluatee department varchar(255)
#     evaluatee_number_of_sentiments: Evaluatee number of sentiments integer
#     evaluatee_positive_sentiments_percentage: Evaluatee positive sentiments percentage float
#     evaluatee_negative_sentiments_percentage: Evaluatee negative sentiments percentage float
#     evaluatee_share: Evaluatee share float
#     evaluatee_course_code: Evaluatee course code varchar(255)
#     """
#
#     __tablename__ = 'csv_professor_sentiment_test'
#     csv_professor_sentiment_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
#     csv_id = db.Column(db.BigInteger)
#     professor = db.Column(db.String(255))
#     evaluatee_overall_sentiment = db.Column(db.Float)
#     evaluatee_department = db.Column(db.String(255))
#     evaluatee_number_of_sentiments = db.Column(db.Integer)
#     evaluatee_positive_sentiments_percentage = db.Column(db.Float)
#     evaluatee_negative_sentiments_percentage = db.Column(db.Float)
#     evaluatee_share = db.Column(db.Float)
#     evaluatee_course_code = db.Column(db.String(255))
#
#     def __repr__(self):
#         return f"CsvProfessorSentiment(csv_professor_sentiment_id={self.csv_professor_sentiment_id}, " \
#                f"csv_id={self.csv_id}, professor={self.professor}, " \
#                f"evaluatee_overall_sentiment={self.evaluatee_overall_sentiment}, " \
#                f"evaluatee_department={self.evaluatee_department}, " \
#                f"evaluatee_number_of_sentiments={self.evaluatee_number_of_sentiments}, " \
#                f"evaluatee_positive_sentiments_percentage={self.evaluatee_positive_sentiments_percentage}, " \
#                f"evaluatee_negative_sentiments_percentage={self.evaluatee_negative_sentiments_percentage}, " \
#                f"evaluatee_share={self.evaluatee_share}, evaluatee_course_code={self.evaluatee_course_code})"
#
#
# class CsvDepartmentSentiment(db.Model):
#     """
#     Csv department sentiment model class attributes
#     csv_department_sentiment_id: Csv department sentiment id number (primary key) (auto increment) bigint
#     csv_id: Csv id number bigint
#     department: Department varchar(255)
#     department_overall_sentiment: Department overall sentiment float
#     department_evaluatee: Department evaluatee varchar(255)
#     department_number_of_sentiments: Department number of sentiments integer
#     department_positive_sentiments_percentage: Department positive sentiments percentage float
#     department_negative_sentiments_percentage: Department negative sentiments percentage float
#     department_share: Department share float
#     """
#
#     __tablename__ = 'csv_department_sentiment_test'
#     csv_department_sentiment_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
#     csv_id = db.Column(db.BigInteger)
#     department = db.Column(db.String(255))
#     department_overall_sentiment = db.Column(db.Float)
#     department_evaluatee = db.Column(db.Integer)
#     department_number_of_sentiments = db.Column(db.Integer)
#     department_positive_sentiments_percentage = db.Column(db.Float)
#     department_negative_sentiments_percentage = db.Column(db.Float)
#     department_share = db.Column(db.Float)
#
#     def __repr__(self):
#         return f"CsvDepartmentSentiment(csv_department_sentiment_id={self.csv_department_sentiment_id}, " \
#                f"csv_id={self.csv_id}, department={self.department}, " \
#                f"department_overall_sentiment={self.department_overall_sentiment}, " \
#                f"department_evaluatee={self.department_evaluatee}, " \
#                f"department_number_of_sentiments={self.department_number_of_sentiments}, " \
#                f"department_positive_sentiments_percentage={self.department_positive_sentiments_percentage}, " \
#                f"department_negative_sentiments_percentage={self.department_negative_sentiments_percentage}, " \
#                f"department_share={self.department_share})"
