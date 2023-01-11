from flask import Flask
from flask_cors import CORS
from sqlalchemy_utils import database_exists, create_database

from config import Config, SQLDatabase, Directories
from extensions import db, bcrypt, mail, session
from matrix.routes.user import user
from matrix.routes.predictalyze import predictalyze
from matrix.routes.dashboard import dashboard

app = Flask(__name__)

# Load all the configuration from the config.py file
app.config.from_object(Config)

CORS(app, supports_credentials=True,
     methods="GET,POST,PUT,DELETE,OPTIONS", expose_headers="Content-Disposition")


db.init_app(app=app)
bcrypt.init_app(app=app)
mail.init_app(app=app)
session.init_app(app=app)

Directories.create_csv_directories()

if not database_exists(SQLDatabase.SQLALCHEMY_DATABASE_URI):
    print(f"Creating database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    create_database(SQLDatabase.SQLALCHEMY_DATABASE_URI)
    print(f"Database created: {app.config['SQLALCHEMY_DATABASE_URI']}")

    with app.app_context():
        db.create_all()

app.register_blueprint(user)
app.register_blueprint(predictalyze)
app.register_blueprint(dashboard)

if __name__ == '__main__':
    app.run()
