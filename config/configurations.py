import os

import mysql.connector
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


# @desc: Email configuration for the Flask app
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
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
     methods="GET,POST,PUT,DELETE,OPTIONS", origins=os.environ.get("DEV_URL"))

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

# @desc: MySQL database connection
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="matrix_lab"
# )

# @desc: The flask sqlalchemy instance
# engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/matrix_lab')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost:3306/matrix_lab' \
                                        '?charset=utf8mb4&collation=utf8mb4_unicode_ci'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if not database_exists(app.config['SQLALCHEMY_DATABASE_URI']):
    print("Creating database")
    create_database(app.config['SQLALCHEMY_DATABASE_URI'])
    print("Database created")

db = SQLAlchemy(app)


# @desc: Config from object method of the Flask app (Should be the last line of the configs)
app.config.from_object(__name__)


# @desc: This is the main method of the Flask app that runs the application.
def run():
    app.run(host='0.0.0.0', debug=True, port=os.environ.get('PORT', 8080))


