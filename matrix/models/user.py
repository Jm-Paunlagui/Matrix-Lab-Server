from extensions import db
from matrix.module import Timezone


class User(db.Model):
    """
    User model class attributes
    user_id: User id number (primary key) (auto increment) bigint
    email: User email address (unique) varchar(255)
    recovery_email: User recovery email address (unique) varchar(255) default "Recovery Email",
    verified_email: User verified email address (unique) varchar(255) default "Unverified Email",
    verified_recovery_email: User verified recovery email address (unique) varchar(255) default "Unverified Email",
    full_name: User full name (unique) varchar(255)
    username: User username (unique) varchar(255)
    password: User password varchar(255)
    role: User role (default: user) varchar(255)
    department: User department (default: None) varchar(255)
    created_at: User created date timestamp
    updated_at: User updated date timestamp
    flag_deleted: User deleted flag (default: 0) tinyint
    flag_locked: User locked flag (default: 0) tinyint
    flag_active: User active flag (default: 0) tinyint
    login_attempts: User login attempts (default: 0) tinyint
    """

    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=True)
    recovery_email = db.Column(db.String(255), unique=True, nullable=True)
    full_name = db.Column(db.String(255), unique=True, nullable=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(255), nullable=False, default="user")
    department = db.Column(db.String(255), nullable=True, default=None)
    created_at = db.Column(db.DateTime, nullable=False,
                           default=Timezone("Asia/Manila").get_timezone_current_time())
    updated_at = db.Column(db.DateTime, nullable=False,
                           default=Timezone("Asia/Manila").get_timezone_current_time())
    flag_deleted = db.Column(db.Boolean, nullable=False, default=False)
    flag_locked = db.Column(db.Boolean, nullable=False, default=False)
    flag_active = db.Column(db.Boolean, nullable=False, default=False)
    login_attempts = db.Column(db.Integer, nullable=False, default=0)
    verified_email = db.Column(
        db.String(255), nullable=True, default="Unverified")
    verified_recovery_email = db.Column(
        db.String(255), nullable=True, default="Unverified")

    def __repr__(self):
        """User model class representation."""
        return f"User('{self.user_id}', '{self.email}', '{self.recovery_email}', '{self.full_name}', " \
               f"'{self.username}', '{self.password}', '{self.role}', '{self.department}', " \
               f"'{self.created_at}', '{self.updated_at}', '{self.flag_deleted}', '{self.flag_locked}', " \
               f"'{self.flag_active}', '{self.login_attempts}', '{self.verified_email}', " \
            f"'{self.verified_recovery_email}')"
