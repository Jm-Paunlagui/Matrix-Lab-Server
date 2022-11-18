from config.configurations import db
from modules.module import Timezone


class CsvModel(db.Model):
    """
    Csv model class attributes
    csv_id: Csv id number (primary key) (auto increment) bigint
    csv_name: Csv name varchar(255)
    csv_question: Csv question varchar(255)
    csv_file_path: Csv file path text
    school_year: School year varchar(255)
    school_semester: School semester varchar(255)
    date_uploaded: Csv date uploaded timestamp
    date_processed: Csv date processed timestamp
    """

    __tablename__ = 'csvs'
    csv_id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    csv_name: str = db.Column(db.String(255), nullable=False)
    csv_question: str = db.Column(db.String(255), nullable=False)
    csv_file_path: str = db.Column(db.Text, nullable=False)
    school_year: str = db.Column(db.String(255), nullable=False)
    school_semester: str = db.Column(db.String(255), nullable=False)
    date_uploaded: str = db.Column(db.DateTime, nullable=False,
                                   default=Timezone("Asia/Manila").get_timezone_current_time())
    date_processed: str = db.Column(db.DateTime, nullable=False,
                                    default=Timezone("Asia/Manila").get_timezone_current_time())

    def __repr__(self):
        """Csv model class representation."""
        return f"CsvModel(csv_id={self.csv_id}, csv_name={self.csv_name}, csv_question={self.csv_question}, " \
               f"csv_file_path={self.csv_file_path}, school_year={self.school_year}, " \
               f"school_semester={self.school_semester}, date_uploaded={self.date_uploaded}, " \
               f"date_processed={self.date_processed})"

    # @desc: For Descending Order (newest to oldest) in the csvs table
    def __lt__(self, other):
        return self.csv_id < other.csv_id

    # @desc: For Ascending Order (oldest to newest) in the csvs table
    def __gt__(self, other):
        return self.csv_id > other.csv_id


class CsvDepartmentModel(db.Model):
    """
    Csv department model class attributes
    csv_id: Csv id number (primary key) (auto increment) bigint
    csv_name: Csv name varchar(255)
    csv_question: Csv question varchar(255)
    csv_file_path: Csv file path text
    school_year: School year varchar(255)
    school_semester: School semester varchar(255)
    date_uploaded: Csv date uploaded timestamp
    date_processed: Csv date processed timestamp
    """

    __tablename__ = 'csvs_department'
    csv_id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    csv_name: str = db.Column(db.String(255), nullable=False)
    csv_question: str = db.Column(db.String(255), nullable=False)
    csv_file_path: str = db.Column(db.Text, nullable=False)
    school_year: str = db.Column(db.String(255), nullable=False)
    school_semester: str = db.Column(db.String(255), nullable=False)
    date_uploaded: str = db.Column(db.DateTime, nullable=False,
                                   default=Timezone("Asia/Manila").get_timezone_current_time())
    date_processed: str = db.Column(db.DateTime, nullable=False,
                                    default=Timezone("Asia/Manila").get_timezone_current_time())

    def __repr__(self):
        """Csv department model class representation."""
        return f"CsvDepartmentModel(csv_id={self.csv_id}, csv_name={self.csv_name}, csv_question={self.csv_question}, " \
               f"csv_file_path={self.csv_file_path}, school_year={self.school_year}, " \
               f"school_semester={self.school_semester}, date_uploaded={self.date_uploaded}, " \
               f"date_processed={self.date_processed})"

    # @desc: For Descending Order (newest to oldest) in the csvs table
    def __lt__(self, other):
        return self.csv_id < other.csv_id

    # @desc: For Ascending Order (oldest to newest) in the csvs table
    def __gt__(self, other):
        return self.csv_id > other.csv_id


class CsvProfessorModel(db.Model):
    """
    Csv professor model class attributes
    csv_id: Csv id number (primary key) (auto increment) bigint
    csv_name: Csv name varchar(255)
    csv_question: Csv question varchar(255)
    csv_file_path: Csv file path text
    school_year: School year varchar(255)
    school_semester: School semester varchar(255)
    date_uploaded: Csv date uploaded timestamp
    date_processed: Csv date processed timestamp
    """

    __tablename__ = 'csvs_professor'
    csv_id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    csv_name: str = db.Column(db.String(255), nullable=False)
    csv_question: str = db.Column(db.String(255), nullable=False)
    csv_file_path: str = db.Column(db.Text, nullable=False)
    school_year: str = db.Column(db.String(255), nullable=False)
    school_semester: str = db.Column(db.String(255), nullable=False)
    date_uploaded: str = db.Column(db.DateTime, nullable=False,
                                   default=Timezone("Asia/Manila").get_timezone_current_time())
    date_processed: str = db.Column(db.DateTime, nullable=False,
                                    default=Timezone("Asia/Manila").get_timezone_current_time())

    def __repr__(self):
        """Csv professor model class representation."""
        return f"CsvProfessorModel(csv_id={self.csv_id}, csv_name={self.csv_name}, csv_question={self.csv_question}, " \
               f"csv_file_path={self.csv_file_path}, school_year={self.school_year}, " \
               f"school_semester={self.school_semester}, date_uploaded={self.date_uploaded}, " \
               f"date_processed={self.date_processed})"

    # @desc: For Descending Order (newest to oldest) in the csvs table
    def __lt__(self, other):
        return self.csv_id < other.csv_id

    # @desc: For Ascending Order (oldest to newest) in the csvs table
    def __gt__(self, other):
        return self.csv_id > other.csv_id


class CsvErrorModel(db.Model):
    """
    Csv error model class attributes
    csv_error_id: Csv error id number (primary key) (auto increment) bigint
    name_of: Name varchar(255)
    csv_error: Csv error text
    date_occurred: Csv error date occurred timestamp
    """

    __tablename__ = 'csvs_error'
    csv_error_id: int = db.Column(
        db.Integer, primary_key=True, autoincrement=True)
    name_of: str = db.Column(db.String(255), nullable=False)
    csv_error: str = db.Column(db.Text, nullable=False)
    date_occurred: str = db.Column(db.DateTime, nullable=False,
                                   default=Timezone("Asia/Manila").get_timezone_current_time())

    def __repr__(self):
        """Csv error model class representation."""
        return f"CsvErrorModel(csv_error_id={self.csv_error_id}, name_of={self.name_of}, csv_error={self.csv_error}, " \
               f"date_occurred={self.date_occurred})"

    # @desc: For Descending Order (newest to oldest) in the csvs_error table
    def __lt__(self, other):
        return self.csv_error_id < other.csv_error_id

    # @desc: For Ascending Order (oldest to newest) in the csvs_error table
    def __gt__(self, other):
        return self.csv_error_id > other.csv_error_id
