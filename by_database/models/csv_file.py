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
    csv_file_path = db.Column(db.Text)
    school_year = db.Column(db.String(255))
    school_semester = db.Column(db.String(255))
    flag_deleted = db.Column(db.Boolean)
    flag_release = db.Column(db.Boolean)

    def __init__(self, csv_question, csv_file_path, school_year, school_semester, flag_deleted, flag_release):
        self.csv_question = csv_question
        self.csv_file_path = csv_file_path
        self.school_year = school_year
        self.school_semester = school_semester
        self.flag_deleted = flag_deleted
        self.flag_release = flag_release

    def __repr__(self):
        return f"CsvModel(csv_id={self.csv_id}, csv_question={self.csv_question}, " \
               f"csv_file_path={self.csv_file_path}, school_year={self.school_year}, " \
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
    csv_id = db.Column(db.BigInteger, primary_key=True, nullable=False)
    evaluatee = db.Column(db.String(255))
    department = db.Column(db.String(255))
    course_code = db.Column(db.String(255))
    sentence = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(255))
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
