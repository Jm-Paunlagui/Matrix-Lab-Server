import os

import redis


class Directories:
    # @desc: The root path of the application
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

    # @desc: The path to the CSV folder
    CSV_FOLDER = os.path.join(ROOT_PATH, "csv_files")

    # @desc: The path to the CSV uploaded folder
    CSV_UPLOADED_FOLDER = os.path.join(CSV_FOLDER, "uploaded_csv_files")

    # @desc: The path to the CSV reformatted folder
    CSV_REFORMATTED_FOLDER = os.path.join(CSV_FOLDER, "reformatted_csv_files")

    # @desc: The path to the CSV analyzed folder
    CSV_ANALYZED_FOLDER = os.path.join(CSV_FOLDER, "analyzed_csv_files")

    # @desc: The path to the CSV department analysis folder
    CSV_DEPARTMENT_ANALYSIS_FOLDER = os.path.join(CSV_FOLDER, "department_analysis_csv_files")

    # @desc: The path to the CSV professor analysis folder
    CSV_PROFESSOR_ANALYSIS_FOLDER = os.path.join(CSV_FOLDER, "professor_analysis_csv_files")

    CSV_USER_COLLECTION_OF_SENTIMENT_PER_EVALUATEE_FOLDER = \
        os.path.join(CSV_FOLDER, "user_collection_of_sentiment_per_evaluatee_csv_files")

    DEEP_LEARNING_MODEL_FOLDER = os.path.join(ROOT_PATH, "matrix/deep_learning_model")

    # @desc: Creates directories for the CSV files if they do not exist
    @staticmethod
    def create_csv_directories():
        if not os.path.exists(Directories.CSV_FOLDER):
            os.mkdir(Directories.CSV_FOLDER)

        if not os.path.exists(Directories.CSV_UPLOADED_FOLDER):
            os.mkdir(Directories.CSV_UPLOADED_FOLDER)

        if not os.path.exists(Directories.CSV_REFORMATTED_FOLDER):
            os.mkdir(Directories.CSV_REFORMATTED_FOLDER)

        if not os.path.exists(Directories.CSV_ANALYZED_FOLDER):
            os.mkdir(Directories.CSV_ANALYZED_FOLDER)

        if not os.path.exists(Directories.CSV_DEPARTMENT_ANALYSIS_FOLDER):
            os.mkdir(Directories.CSV_DEPARTMENT_ANALYSIS_FOLDER)

        if not os.path.exists(Directories.CSV_PROFESSOR_ANALYSIS_FOLDER):
            os.mkdir(Directories.CSV_PROFESSOR_ANALYSIS_FOLDER)

        if not os.path.exists(Directories.CSV_USER_COLLECTION_OF_SENTIMENT_PER_EVALUATEE_FOLDER):
            os.mkdir(Directories.CSV_USER_COLLECTION_OF_SENTIMENT_PER_EVALUATEE_FOLDER)


class AllowedExtensions:
    ALLOWED_EXTENSIONS = {"csv"}


class FlaskEmail:
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 465
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")


class FlaskRedis:
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_REDIS = redis.from_url("redis://127.0.0.1:6379")


class SQLDatabase:
    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class SecretKeys:
    PUBLIC_KEY = b"-----BEGIN PUBLIC KEY-----\n" + \
                 os.environ.get("MATRIX_RSA_PUBLIC_KEY").encode() + \
                 b"\n-----END PUBLIC KEY-----"
    PRIVATE_KEY = b"-----BEGIN PRIVATE KEY-----\n" + \
                  os.environ.get("MATRIX_RSA_PRIVATE_KEY").encode() + \
                  b"\n-----END PRIVATE KEY-----"
    SECRET_KEY_B32 = os.environ.get("SECRET_KEY_BASE32")
    SECRET_KEY = os.environ.get("SECRET_KEY")


class Config:
    DEBUG = True
    TESTING = False
    # Reuse the class variables from the above classes to avoid repetition of code and to make the code more readable
    ALLOWED_EXTENSIONS = AllowedExtensions.ALLOWED_EXTENSIONS
    CSV_FOLDER = Directories.CSV_FOLDER
    CSV_UPLOADED_FOLDER = Directories.CSV_UPLOADED_FOLDER
    CSV_REFORMATTED_FOLDER = Directories.CSV_REFORMATTED_FOLDER
    DEEP_LEARNING_MODEL_FOLDER = Directories.DEEP_LEARNING_MODEL_FOLDER

    MAIL_SERVER = FlaskEmail.MAIL_SERVER
    MAIL_PORT = FlaskEmail.MAIL_PORT
    MAIL_USE_TLS = FlaskEmail.MAIL_USE_TLS
    MAIL_USE_SSL = FlaskEmail.MAIL_USE_SSL
    MAIL_USERNAME = FlaskEmail.MAIL_USERNAME
    MAIL_PASSWORD = FlaskEmail.MAIL_PASSWORD

    SESSION_TYPE = FlaskRedis.SESSION_TYPE
    SESSION_PERMANENT = FlaskRedis.SESSION_PERMANENT
    SESSION_USE_SIGNER = FlaskRedis.SESSION_USE_SIGNER
    SESSION_REDIS = FlaskRedis.SESSION_REDIS

    SQLALCHEMY_DATABASE_URI = SQLDatabase.SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = SQLDatabase.SQLALCHEMY_TRACK_MODIFICATIONS

    PUBLIC_KEY = SecretKeys.PUBLIC_KEY
    PRIVATE_KEY = SecretKeys.PRIVATE_KEY
    SECRET_KEY_B32 = SecretKeys.SECRET_KEY_B32
    SECRET_KEY = SecretKeys.SECRET_KEY


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
