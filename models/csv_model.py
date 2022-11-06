from config.configurations import db
from modules.module import Timezone


class CsvModel(db.Model):
    """
    Csv model class attributes
    csv_id: Csv id number (primary key) (auto increment) bigint
    csv_name: Csv name varchar(255)
    csv_file_path: Csv file path text
    date_uploaded: Csv date uploaded timestamp
    date_processed: Csv date processed timestamp
    """

    __tablename__ = 'csvs'
    csv_id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    csv_name: str = db.Column(db.String(255), nullable=False)
    csv_file_path: str = db.Column(db.Text, nullable=False)
    date_uploaded: str = db.Column(db.DateTime, nullable=False,
                                   default=Timezone("Asia/Manila").get_timezone_current_time())
    date_processed: str = db.Column(db.DateTime, nullable=False,
                                    default=Timezone("Asia/Manila").get_timezone_current_time())

    def __repr__(self):
        """Csv model class representation."""
        return f"CsvModel('{self.csv_id}', '{self.csv_name}', '{self.csv_file_path}', '{self.date_uploaded}', " \
               f"'{self.date_processed}')"
