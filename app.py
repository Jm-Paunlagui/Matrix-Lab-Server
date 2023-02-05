from flask import Flask, jsonify
from flask_cors import CORS
from sqlalchemy_utils import create_database, database_exists

from config import Config, Directories, SQLDatabase
from extensions import bcrypt, db, mail
from matrix.models.csv_file import (CsvAnalyzedSentiment, CsvCourses,
                                    CsvDepartmentSentiment, CsvModelDetail,
                                    CsvProfessorSentiment, CsvTimeElapsed,
                                    ErrorModel)
from matrix.models.user import User
from matrix.routes.dashboard import dashboard
from matrix.routes.predictalyze import predictalyze
from matrix.routes.user import user

app = Flask(__name__)

# Load all the configuration from the config.py file
app.config.from_object(Config)

CORS(app, supports_credentials=True,
     methods="GET,POST,PUT,DELETE,OPTIONS", expose_headers="Content-Disposition")

db.init_app(app=app)
bcrypt.init_app(app=app)
mail.init_app(app=app)

Directories.create_csv_directories()

# if not database_exists(SQLDatabase.SQLALCHEMY_DATABASE_URI):
#     print(f"Creating database: {app.config['SQLALCHEMY_DATABASE_URI']}")
#     create_database(SQLDatabase.SQLALCHEMY_DATABASE_URI)
#     print(f"Database created: {app.config['SQLALCHEMY_DATABASE_URI']}")

# with app.app_context():
#     db.create_all()
#
#     hashed_password = bcrypt.generate_password_hash("123@Qwer").decode("utf-8")
#
#     default_user = User(
#         email="paunlagui.jm.cs@gmail.com",
#         full_name="John Moises E. Paunlagui",
#         username="admin-jm",
#         password=hashed_password,
#         role="admin",
#         verified_email="Verified"
#     )
#
#     db.session.add(default_user)
#     db.session.commit()

print("Default user created")

app.register_blueprint(user)
app.register_blueprint(predictalyze)
app.register_blueprint(dashboard)


@app.route("/", methods=["GET"])
def index():
    # Github: Jm-Paunlagui
    return jsonify({"message": "Welcome to the Matrix LAB API, made by John Moises E. Paunlagui. "
                               "Github: Jm-Paunlagui"}), 200


@app.route("/made-with", methods=["GET"])
def model_details():
    details = [
        {
            "model_name": "model.h5",
            "model_size": "20 MB",
            "model_total_parameters": "1,723,777",
            "model_non_trainable_parameters": "0",
            "model_total_layers": "6",
            "model_loss": "binary_crossentropy",
            "model_optimizer": "Adam",
            "model_metrics": "Mean, MeanMetricWrapper",
        }
    ]
    tools_n_tech = [
        {
            "name": "Python",
            "version": "3.10.5",
            "link": "https://www.python.org/",
            "description":
                "Python is a programming language that lets you work quickly and integrate systems more effectively. Python is easy to learn and has a large body of libraries and tools that let you create your own applications.",
        },
        {
            "name": "TensorFlow",
            "version": "2.9.1",
            "link": "https://www.tensorflow.org/",
            "description":
                "TensorFlow is an open source machine learning platform for everyone. It is used to build and train neural networks.",
        },
        {
            "name": "Keras",
            "version": "2.9.0",
            "link": "https://keras.io/",
            "description":
                "Keras is a high-level neural network library for the Python programming language. ",
        },
        {
            "name": "Flask",
            "version": "2.1.2",
            "link": "https://flask.palletsprojects.com/en/1.1.x/",
            "description":
                "Flask is a microframework for Python based on Werkzeug and Jinja2. It is designed to be as simple as possible while still being powerful enough to be useful.",
        },
        {
            "name": "MariaDB",
            "version": "10.4.24",
            "link": "https://mariadb.org/",
            "description":
                "MariaDB is a free and open-source relational database management system (RDBMS) based on the MySQL server. ",
        },
        {
            "name": "MySQL",
            "version": "5.0.37",
            "link": "https://www.mysql.com/",
            "description":
                "MySQL is a relational database management system (RDBMS) that is used to create, manage, and administer databases.",
        },
        {
            "name": "phpMyAdmin",
            "version": "5.2.0",
            "link": "https://www.phpmyadmin.net/",
            "description":
                "phpMyAdmin is a free software web application for MySQL and MariaDB. It is a fully featured, easy to use, and fully standards-compliant web based MySQL database management system (DBMS) with a focus on ease of use and user friendlyness.",
        },
        {
            "name": "Visual Studio Code",
            "version": "1.70.2",
            "link": "https://code.visualstudio.com/",
            "description":
                "Visual Studio Code is a free and open-source code editor for the Microsoft Windows platform. ",
        },
        {
            "name": "PyCharm Professional Edition",
            "version": "2022.2.1",
            "link": "https://www.jetbrains.com/pycharm/",
            "description":
                "PyCharm is the best IDE for Python. It is a cross-platform IDE that supports Python development on Windows, macOS, and Linux.",
        },
        {
            "name": "WebStorm Professional Edition",
            "version": "2022.2.1",
            "link": "https://www.jetbrains.com/webstorm/",
            "description":
                "WebStorm is the smartest JavaScript IDE by JetBrains. It provides code completion, on-the-fly error detection, and refactoring for JavaScript, TypeScript, and JSX.",
        },
        {
            "name": "XAMPP",
            "version": "8.1.6",
            "link": "https://www.apachefriends.org/en/xampp.html",
            "description":
                "XAMPP is a free web server software package that allows you to develop and test your web applications without installing any software on your computer.",
        },
        {
            "name": "Git",
            "version": "2.37.3",
            "link": "https://git-scm.com/",
            "description":
                "Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency. ",
        },
        {
            "name": "GitHub",
            "version": "",
            "link": "https://github.com/",
            "description":
                "GitHub is a web-based hosting service for version control using Git.",
        },
        {
            "name": "Visual Studio 2022 Desktop Development with C++",
            "version": "17.3.3",
            "link": "https://www.microsoft.com/en-us/download/details.aspx?id=50000",
            "description":
                "Visual Studio is a free and open-source software development platform designed to make it easier to create, manage, and share software projects. ",
        },
        {
            "name": "Github Copilot",
            "version": "",
            "link": "https://github.com/features/copilot",
            "description":
                "GitHub Copilot is a AI-powered software development platform that helps developers build software faster and with less effort, supported on many Text Editors and IDEs.",
        },
        {
            "name": "Postman",
            "version": "9.30.1",
            "link": "https://www.getpostman.com/",
            "description":
                "Postman is a web application that allows you to create and manage RESTful APIs in a single place and share them with anyone.",
        },
        {
            "name": "CUDA",
            "version": "11.7.1",
            "link": "https://developer.nvidia.com/cuda-toolkit",
            "description":
                "CUDA is a parallel computing platform and runtime library developed by NVIDIA to accelerate computing in general and deep learning in particular.",
        },
        {
            "name": "cuDNN",
            "version": "8.4.1",
            "link": "https://developer.nvidia.com/cudnn",
            "description":
                "cuDNN is a library that provides high-performance, memory-efficient convolutional neural network libraries, accelerates the performance of deep neural networks.",
        },
    ]

    return jsonify({"model_details": details, "tools_and_technologies": tools_n_tech})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, load_dotenv=True)
