from config.configurations import db
from modules.module import Timezone


# desc: User model class
class User(db.Model):
    """
    User model class attributes
    user_id: User id number (primary key) (auto increment) bigint
    email: User email address (unique) varchar(255)
    secondary_email: User secondary email address (unique) varchar(255) default "Secondary Email"
    recovery_email: User recovery email address (unique) varchar(255) default "Recovery Email"
    full_name: User full name (unique) varchar(255)
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
    user_id: int = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email: str = db.Column(db.String(255), unique=True, nullable=True)
    secondary_email: str = db.Column(
        db.String(255), unique=True, nullable=True)
    recovery_email: str = db.Column(db.String(255), unique=True, nullable=True)
    full_name: str = db.Column(db.String(255), unique=True, nullable=True)
    username: str = db.Column(db.String(255), unique=True, nullable=False)
    password: str = db.Column(db.String(255), nullable=True)
    role: str = db.Column(db.String(255), nullable=False, default="user")
    created_at: str = db.Column(db.DateTime, nullable=False,
                                default=Timezone("Asia/Manila").get_timezone_current_time())
    updated_at: str = db.Column(db.DateTime, nullable=False,
                                default=Timezone("Asia/Manila").get_timezone_current_time())
    flag_deleted = db.Column(db.Boolean, nullable=False, default=False)
    password_reset_token = db.Column(db.Text, nullable=True)

    def __repr__(self):
        """User model class representation."""
        return f"User('{self.user_id}', '{self.email}', '{self.secondary_email}', '{self.recovery_email}', " \
               f"'{self.full_name}', '{self.username}', '{self.password}', '{self.role}', '{self.created_at}', " \
               f"'{self.updated_at}', '{self.flag_deleted}', '{self.password_reset_token}')"
