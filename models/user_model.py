from config.configurations import db
from modules.datetime_tz import timezone_current_time


# desc: User model class
class User(db.Model):
    """
    User model class attributes
    user_id: User id number (primary key) (auto increment) bigint
    email: User email address (unique) varchar(255)
    first_name: User first name varchar(255)
    last_name: User last name varchar(255)
    username: User username (unique) varchar(255)
    password: User password varchar(255)
    role: User role (default: user) varchar(255)
    created_at: User created date timestamp
    updated_at: User updated date timestamp
    flag_deleted: User deleted flag (default: 0) tinyint
    password_reset_token: User password reset token text
    security_code: User security code varchar(255)
    """

    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    recovery_email = db.Column(db.String(255), unique=True, nullable=True)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(255), nullable=False, default='user')
    created_at = db.Column(db.DateTime, nullable=False,
                           default=timezone_current_time)
    updated_at = db.Column(db.DateTime, nullable=False,
                           default=timezone_current_time)
    flag_deleted = db.Column(db.Boolean, nullable=False, default=False)
    password_reset_token = db.Column(db.Text, nullable=True)
    security_code = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        """User model class representation."""
        return f"User('{self.user_id}', '{self.email}', '{self.recovery_email}', '{self.first_name}', " \
               f"'{self.last_name}', '{self.username}', '{self.password}', '{self.role}', '{self.created_at}', " \
               f"'{self.updated_at}', '{self.flag_deleted}', '{self.password_reset_token}', '{self.security_code}')"
