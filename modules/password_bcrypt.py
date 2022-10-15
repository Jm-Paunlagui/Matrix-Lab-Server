# @desc: Password text generator for the user's password
import random
import string

from config.configurations import app
from flask_bcrypt import Bcrypt

from modules.input_validation import validate_password

# @desc: The bcrypt instance
bcrypt = Bcrypt(app)


def password_generator():
    """Password generator function with a length of 15 characters."""
    password_length = 15
    special_characters = "#?!@$%^&*-"
    password_characters = string.ascii_letters + string.digits + special_characters
    passwords = ''.join(random.choices(password_characters, k=password_length))
    if validate_password(passwords):
        return passwords
    return password_generator()


def password_hash_check(hashed_password: str, password: str):
    """Check if the password is correct and return a boolean value."""
    if not bcrypt.check_password_hash(hashed_password, password):
        return False
    return True


def password_hasher(password: str):
    """Hash the password and return the hashed password."""
    return bcrypt.generate_password_hash(password)
