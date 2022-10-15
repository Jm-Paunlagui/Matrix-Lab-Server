# @desc: Password text generator for the user's password
import random
import string

from config.configurations import app
from flask_bcrypt import Bcrypt

from modules.input_validation import validate_password

# @desc: The bcrypt instance
bcrypt = Bcrypt(app)


def password_generator():
    # @desc: Password length
    password_length = 15

    # @desc: Custom special characters for the password
    special_characters = "#?!@$%^&*-"

    # @desc: Password characters
    password_characters = string.ascii_letters + string.digits + special_characters

    # @desc: Generate the password should be 8 characters long and alphanumeric and special characters
    passwords = ''.join(random.choices(password_characters, k=password_length))

    # @desc: Check if the password is valid
    if validate_password(passwords):
        return passwords
    else:
        return password_generator()


# @desc: Password hasher checker function
def password_hash_check(hashed_password: str, password: str):
    if not bcrypt.check_password_hash(hashed_password, password):
        return False
    else:
        return True


# @desc: Password hasher function
def password_hasher(password: str):
    hashed_password = bcrypt.generate_password_hash(password)
    return hashed_password
