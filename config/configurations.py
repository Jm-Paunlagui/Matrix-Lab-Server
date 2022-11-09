import os
import socket

import redis
from flask import Flask
from flask_cors import CORS
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_utils import database_exists, create_database

# Create the Flask app and load the config file
app = Flask(__name__)

# @desc: This method pushes the application context to the top of the stack.
app.app_context().push()

# @desc: Root path of the application (the directory where the application is located)
app.config["ROOT_PATH"] = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# @desc: CSV file upload path configuration for the Flask app
app.config["CSV_UPLOADED_FOLDER"] = os.path.join(
    app.config["ROOT_PATH"], "csv_files\\uploaded_csv_files")
# @desc: CSV file reformatted path configuration for the Flask app
app.config["CSV_REFORMATTED_FOLDER"] = os.path.join(
    app.config["ROOT_PATH"], "csv_files\\reformatted_csv_files")
# @desc: CSV file analyzed path configuration for the Flask app
app.config["CSV_ANALYZED_FOLDER"] = os.path.join(
    app.config["ROOT_PATH"], "csv_files\\analyzed_csv_files")
app.config["DEEP_LEARNING_MODEL_FOLDER"] = os.path.join(
    app.config["ROOT_PATH"], "deep_learning_model")
app.config["ALLOWED_EXTENSIONS"] = {"csv"}

# @desc: Email configuration for the Flask app
app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_USE_SSL"] = True
app.config["MAIL_USE_TLS"] = False
app.config["MAIL_PORT"] = 465
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")

# @desc: Secret key of the application
app.secret_key = os.environ.get("SECRET_KEY")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

# @desc: Cross-Origin Resource Sharing configuration for the Flask app to allow requests from the client
CORS(app, supports_credentials=True,
     methods="GET,POST,PUT,DELETE,OPTIONS")

# @desc: The flask mail instance
mail = Mail(app)

# @desc: RSA keys for JWT
private_key = b"-----BEGIN PRIVATE KEY-----\n" + \
              os.environ.get("MATRIX_RSA_PRIVATE_KEY").encode() + \
              b"\n-----END PRIVATE KEY-----"
public_key = b"-----BEGIN PUBLIC KEY-----\n" + \
             os.environ.get("MATRIX_RSA_PUBLIC_KEY").encode() + \
             b"\n-----END PUBLIC KEY-----"

# @desc: The redis configuration for the Flask app
SESSION_TYPE = "redis"
SESSION_PERMANENT = False
SESSION_USE_SIGNER = True
SESSION_REDIS = redis.from_url("redis://127.0.0.1:6379")

# @desc: The flask sqlalchemy instance
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    "SQLALCHEMY_DATABASE_URI")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if not database_exists(app.config['SQLALCHEMY_DATABASE_URI']):
    print("Creating database")
    create_database(app.config['SQLALCHEMY_DATABASE_URI'])
    print("Database created")

db = SQLAlchemy(app)
# noinspection PyUnresolvedReferences
# from models import user_model, csv_model
from models import user_model, csv_model
db.create_all()

# @desc: Config from object method of the Flask app (Should be the last line of the configs)
app.config.from_object(__name__)


def run():
    """
    Audit: Binding to all interfaces detected with hardcoded values - BAN-B104 - Severity: Major
    Binding to all network interfaces can potentially open up a service to traffic on unintended interfaces,
    that may not be properly documented or secured. This can be prevented by changing the code, so it explicitly only
    allows access from localhost

    When binding to 0.0.0.0, you accept incoming connections from anywhere. During development, an application may have
    security vulnerabilities making it susceptible to SQL injections and other attacks. Therefore, when the application
    is not ready for production, accepting connections from anywhere can be dangerous.

    It is recommended to use 127.0.0.1 or local host during development phase. This prevents others from targeting your
    application and executing SQL injections against your project. - DeepSource
    """
    # @desc: Recommended to use
    # Create a UDP socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 31137))  # Bind to localhost
