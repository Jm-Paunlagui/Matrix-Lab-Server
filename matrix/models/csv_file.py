from extensions import db
from matrix.module import Timezone


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

    __tablename__ = 'csv_model_detail'
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

    __tablename__ = 'csv_analyzed_sentiment'
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


class CsvCourses(db.Model):
    """
    Csv courses model class attributes
    csv_courses_id: Csv courses id number (primary key) (auto increment) bigint
    csv_id: Csv id number (foreign key) (not null) bigint relationship with csv model
    course_code: Course code varchar(255)
    course_for_name: Course for name varchar(255)
    course_for_department: Course for department varchar(255)
    number_of_responses: Number of responses integer
    """
    __tablename__ = 'csv_courses'
    csv_courses_id = db.Column(
        db.BigInteger, primary_key=True, autoincrement=True)
    csv_id = db.Column(db.BigInteger, nullable=False)
    course_code = db.Column(db.String(255))
    course_for_name = db.Column(db.String(255))
    course_for_department = db.Column(db.String(255))
    number_of_responses = db.Column(db.Integer)

    def __repr__(self):
        return f"CsvCourses(csv_courses_id={self.csv_courses_id}, csv_id={self.csv_id}, " \
               f"course_code={self.course_code}, course_for_name={self.course_for_name}, " \
               f"course_for_department={self.course_for_department}, number_of_responses={self.number_of_responses})"


class CsvProfessorSentiment(db.Model):
    """
    Csv professor sentiment model class attributes
    csv_professor_sentiment_id: Csv professor sentiment id number (primary key) (auto increment) bigint
    csv_id: Csv id number bigint
    professor: Professor varchar(255) evaluatee
    evaluatee_department: Evaluatee department varchar(255)
    evaluatee_number_of_sentiments: Evaluatee number of sentiments integer
    evaluatee_positive_sentiments_percentage: Evaluatee positive sentiments percentage float
    evaluatee_negative_sentiments_percentage: Evaluatee negative sentiments percentage float
    evaluatee_share: Evaluatee share float
    """

    __tablename__ = 'csv_professor_sentiment'
    csv_professor_sentiment_id = db.Column(
        db.BigInteger, primary_key=True, autoincrement=True)
    csv_id = db.Column(db.BigInteger)
    professor = db.Column(db.String(255))
    evaluatee_department = db.Column(db.String(255))
    evaluatee_number_of_sentiments = db.Column(db.Integer)
    evaluatee_positive_sentiments_percentage = db.Column(db.Float)
    evaluatee_negative_sentiments_percentage = db.Column(db.Float)
    evaluatee_share = db.Column(db.Float)

    def __repr__(self):
        return f"CsvProfessorSentiment(csv_professor_sentiment_id={self.csv_professor_sentiment_id}, " \
               f"csv_id={self.csv_id}, professor={self.professor}, evaluatee_department={self.evaluatee_department}, " \
               f"evaluatee_number_of_sentiments={self.evaluatee_number_of_sentiments}, " \
               f"evaluatee_positive_sentiments_percentage={self.evaluatee_positive_sentiments_percentage}, " \
               f"evaluatee_negative_sentiments_percentage={self.evaluatee_negative_sentiments_percentage}, " \
               f"evaluatee_share={self.evaluatee_share})"


class CsvDepartmentSentiment(db.Model):
    """
    Csv department sentiment model class attributes
    csv_department_sentiment_id: Csv department sentiment id number (primary key) (auto increment) bigint
    csv_id: Csv id number bigint
    department: Department varchar(255)
    department_evaluatee: Department evaluatee varchar(255)
    department_number_of_sentiments: Department number of sentiments integer
    department_positive_sentiments_percentage: Department positive sentiments percentage float
    department_negative_sentiments_percentage: Department negative sentiments percentage float
    department_share: Department share float
    """

    __tablename__ = 'csv_department_sentiment'
    csv_department_sentiment_id = db.Column(
        db.BigInteger, primary_key=True, autoincrement=True)
    csv_id = db.Column(db.BigInteger)
    department = db.Column(db.String(255))
    department_evaluatee = db.Column(db.Integer)
    department_number_of_sentiments = db.Column(db.Integer)
    department_positive_sentiments_percentage = db.Column(db.Float)
    department_negative_sentiments_percentage = db.Column(db.Float)
    department_share = db.Column(db.Float)

    def __repr__(self):
        return f"CsvDepartmentSentiment(csv_department_sentiment_id={self.csv_department_sentiment_id}, " \
               f"csv_id={self.csv_id}, department={self.department}, department_evaluatee={self.department_evaluatee}, " \
               f"department_number_of_sentiments={self.department_number_of_sentiments}, " \
               f"department_positive_sentiments_percentage={self.department_positive_sentiments_percentage}, " \
               f"department_negative_sentiments_percentage={self.department_negative_sentiments_percentage}, " \
               f"department_share={self.department_share})"


class ErrorModel(db.Model):
    """
    Csv error model class attributes
    error_id: Csv error id number (primary key) (auto increment) bigint
    category_error Type of error varchar(255)
    cause_of: Name varchar(255)
    error_type: Csv error text
    date_occurred: Csv error date occurred timestamp
    """

    __tablename__ = 'error_dump'
    error_id: int = db.Column(
        db.Integer, primary_key=True, autoincrement=True)
    category_error: str = db.Column(db.String(255))
    cause_of: str = db.Column(db.String(255), nullable=False)
    error_type: str = db.Column(db.Text, nullable=False)
    date_occurred: str = db.Column(db.DateTime, nullable=False,
                                   default=Timezone("Asia/Manila").get_timezone_current_time())

    def __repr__(self):
        """Csv error model class representation."""
        return f"ErrorModel(error_id={self.error_id}, type_of_error={self.category_error}, " \
               f"cause_of={self.cause_of}, error_type={self.error_type}, date_occurred={self.date_occurred})"


class CsvTimeElapsed(db.Model):
    """
    Csv Time Elapsed Model class attributes
    csv_id: Csv id number
    date_processed: Csv date processed varchar(255)
    time_elapsed: Time elapsed varchar(255)
    pre_formatter_time: Pre formatter time varchar(255)
    post_formatter_time: Post formatter time varchar(255)
    tokenizer_time: Tokenizer time varchar(255)
    padding_time: Padding time varchar(255)
    model_time: Model time varchar(255)
    prediction_time: Prediction time varchar(255)
    sentiment_time: Sentiment time varchar(255)
    adding_predictions_time: Adding predictions time varchar(255)
    adding_to_db: Adding to db time varchar(255)
    analysis_user_time: Analysis user time varchar(255)
    analysis_department_time: Analysis department time varchar(255)
    analysis_collection_time: Analysis collection time varchar(255)
    """

    __tablename__ = 'csvs_time_elapsed'
    csv_id: int = db.Column(db.Integer, primary_key=True)
    date_processed: str = db.Column(db.String(255), nullable=False)
    time_elapsed: str = db.Column(db.String(255), nullable=False)
    pre_formatter_time: str = db.Column(db.String(255), nullable=False)
    post_formatter_time: str = db.Column(db.String(255), nullable=False)
    tokenizer_time: str = db.Column(db.String(255), nullable=False)
    padding_time: str = db.Column(db.String(255), nullable=False)
    model_time: str = db.Column(db.String(255), nullable=False)
    prediction_time: str = db.Column(db.String(255), nullable=False)
    sentiment_time: str = db.Column(db.String(255), nullable=False)
    adding_predictions_time: str = db.Column(db.String(255), nullable=False)
    adding_to_db: str = db.Column(db.String(255), nullable=False)
    analysis_user_time: str = db.Column(db.String(255), nullable=False)
    analysis_department_time: str = db.Column(db.String(255), nullable=False)
    analysis_collection_time: str = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        """Csv Time Elapsed Model class representation."""
        return f"CsvTimeElapsed(csv_id={self.csv_id}, date_processed={self.date_processed}, " \
               f"time_elapsed={self.time_elapsed}, pre_formatter_time={self.pre_formatter_time}, " \
               f"post_formatter_time={self.post_formatter_time}, tokenizer_time={self.tokenizer_time}, " \
               f"padding_time={self.padding_time}, model_time={self.model_time}, " \
               f"prediction_time={self.prediction_time}, sentiment_time={self.sentiment_time}, " \
               f"adding_predictions_time={self.adding_predictions_time}, adding_to_db={self.adding_to_db}, " \
               f"analysis_user_time={self.analysis_user_time}, " \
               f"analysis_department_time={self.analysis_department_time}, " \
               f"analysis_collection_time={self.analysis_collection_time})"
