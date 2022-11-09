import os
import re
from datetime import datetime

import pyotp
import pytz
import random
import string
from config.configurations import app, private_key, public_key
from flask_bcrypt import Bcrypt
import jwt
from flask import request
from ua_parser import user_agent_parser
from werkzeug.user_agent import UserAgent
from werkzeug.utils import cached_property

# @desc: The bcrypt instance
bcrypt = Bcrypt(app)


class AllowedFile:
    """The allowed file class."""

    def __init__(self, filename: str):
        self.filename = filename

    def allowed_file(self) -> bool:
        """
        Check if the file is allowed to be uploaded.

        :return: True if the file is allowed, False otherwise
        """
        return '.' in self.filename and \
               self.filename.rsplit('.', 1)[1].lower(
               ) in app.config["ALLOWED_EXTENSIONS"]

    def secure_filename(self) -> str:
        """
        Secure the filename.

        :return: The secure filename
        """
        return re.sub(r'[^a-zA-Z0-9_.]', '', self.filename)


class Timezone:
    """Get the current time in the local timezone."""

    def __init__(self, timezone: str = None):
        self.timezone = timezone

    def get_timezone_current_time(self):
        """
        Return the current time in the local timezone.

        :return: The current time in the local timezone
        """
        timezone = pytz.timezone(self.timezone)
        return timezone.localize(datetime.now())


class InputFileValidation:
    """Validate the input file."""

    def __init__(self, file):
        self.file = file

    def validate_file(self):
        """
        Validate the input file.

        :return: True if the file is valid, False otherwise
        """
        if self.file.filename == '':
            return False
        if not AllowedFile(self.file.filename).allowed_file():
            return False
        return True


class TextPreprocessing:
    """Text preprocessing class."""

    def __init__(self, text: str):
        self.text = text

    def remove_punctuation(self) -> str:
        """
        Remove punctuation from the text.

        :return: The text without punctuation
        """
        return re.sub(r'[^\w\s]', '', self.text)

    def remove_numbers(self) -> str:
        """
        Remove numbers from the text.

        :return: The text without numbers
        """
        return re.sub(r'\d+', '', self.text)

    def remove_non_ascii_characters(self) -> str:
        """
        Remove non-ASCII characters from the text.

        :return: The text without non-ASCII characters
        """
        return self.text.encode("ascii", "ignore").decode()

    def remove_tabs_carriage_newline(self) -> str:
        """
        Remove tabs, carriage and newline characters from the text.

        :return: The text without tabs, carriage and newline characters
        """
        return re.sub(r'[\t\r\n]', '', self.text)

    def remove_whitespace(self) -> str:
        """
        Remove whitespace from the text.

        :return: The text without whitespace
        """
        return " ".join(self.text.split())

    def remove_special_characters(self) -> str:
        """
        Remove special characters from the text.

        :return: The text without special characters
        """
        return re.sub(r'[^\w\s]', '', self.text)

    def remove_multiple_whitespaces(self) -> str:
        """
        Remove multiple whitespaces from the text.

        :return: The text without multiple whitespaces
        """
        return re.sub(r'\s+', ' ', self.text)

    def remove_urls(self) -> str:
        """
        Remove URLs from the text.

        :return: The text without URLs
        """
        return re.sub(r'http\S+', '', self.text)

    def remove_emails(self) -> str:
        """
        Remove emails from the text.

        :return: The text without emails
        """
        return re.sub(r'\S+@\S+', '', self.text)

    def remove_html_tags(self) -> str:
        """
        Remove HTML tags

        :return: The text without HTML tags
        """
        return re.sub(r'<.*?>', '', self.text)

    def remove_whitespace_at_beginning_and_end(self) -> str:
        """
        Remove whitespace at the beginning and end of the text.

        :return: The text without whitespace at the beginning and end
        """
        return self.text.strip()

    def clean_text(self) -> str:
        """
        Clean the text.

        :return: The cleaned text
        """
        self.text = self.remove_punctuation()
        self.text = self.remove_numbers()
        self.text = self.remove_non_ascii_characters()
        self.text = self.remove_tabs_carriage_newline()
        self.text = self.remove_whitespace()
        self.text = self.remove_special_characters()
        self.text = self.remove_multiple_whitespaces()
        self.text = self.remove_urls()
        self.text = self.remove_emails()
        self.text = self.remove_html_tags()
        self.text = self.remove_whitespace_at_beginning_and_end()
        return self.text


class InputTextValidation:
    """Validate user text input"""

    def __init__(self, user_input: str = None):
        self.user_input = user_input

    @staticmethod
    def validate_empty_fields(*args: str):
        """
        Checks if any of the fields are empty.

        :param args: The fields to check
        """
        return all(not arg == "" or arg is None or arg == " " for arg in args)

    def validate_email(self):
        """
        Checks if the email is valid.

        :return: True if the email is valid, False otherwise
        """
        return bool(re.compile(r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@(["
                               r"-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])")
                    .match(self.user_input))

    def validate_password(self):
        """
        Checks if the password is valid.

        :return: True if the password is valid, False otherwise
        """
        return bool(re.compile(r"^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$")
                    .match(self.user_input))

    def validate_username(self):
        """
        Checks if the username is valid.

        :return: True if the username is valid, False otherwise
        """
        return bool(re.compile(r"^[a-zA-Z0-9_-]{5,20}$").match(self.user_input))

    def validate_text(self):
        """
        Checks if the text is valid.

        :return: True if the text is valid, False otherwise
        """
        return bool(re.compile(r"^[^0-9_!¡?÷¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}$").match(self.user_input))

    def validate_number(self):
        """
        Checks if the number is valid.

        :return: True if the number is valid, False otherwise
        """
        return bool(re.compile(r"^[0-9]+$").match(str(self.user_input)))

    def validate_question(self):
        """
        Checks if the question is valid.

        :return: True if the question is valid, False otherwise
        """
        return bool(re.compile(r"^[^0-9_!¡?÷¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}$").match(self.user_input))

    def validate_school_year(self):
        """
        Checks if the school year is valid.

        :return: True if the school year is valid, False otherwise
        """
        return bool(re.compile(r"^(S\.Y\. )\d{4}-\d{4}$").match(self.user_input))


class PasswordBcrypt:
    """Hash and check password."""

    def __init__(self, password: str = None):
        self.password = password

    def password_generator(self):
        """
        Password generator function with a length of 15 characters.

        :return: The generated password
        """
        password_length = 15
        special_characters = "#?!@$%^&*-"
        password_characters = string.ascii_letters + string.digits + special_characters
        passwords = ''.join(random.choices(
            password_characters, k=password_length))
        if InputTextValidation(passwords).validate_password():
            return passwords
        return self.password_generator()

    def password_hasher(self):
        """
        Hash the password and return the hashed password.

        :return: The hashed password
        """
        return bcrypt.generate_password_hash(self.password)

    def password_hash_check(self, password_hash):
        """
        Check if the password is correct and return a boolean value.

        :param password_hash: The hashed password
        :return: True if the password is correct, False otherwise
        """
        return bcrypt.check_password_hash(password_hash, self.password)


class PayloadSignature:
    """Encode and decode payload with private and public key."""

    def __init__(self, encoded: str = None, payload: dict = None):
        self.encoded = encoded
        self.payload = payload

    def encode_payload(self):
        """
        Encode payload with private key.

        :return: The encoded payload
        """
        return jwt.encode(self.payload, private_key, algorithm="RS256")

    def decode_payload(self):
        """
        Decode payload with public key.

        :return: The decoded payload
        """
        return jwt.decode(self.encoded, public_key, algorithms=["RS256"], verify=True)


class ToptCode:
    """Generate and verify 2FA code."""
    totp = pyotp.TOTP(os.getenv("SECRET_KEY_BASE32"), digits=7)

    @staticmethod
    def topt_code():
        """
        Generates a 6-digit code for 2FA. The code expires after 30 seconds.

        :return: The generated 2FA code
        """
        return ToptCode.totp.now()

    @staticmethod
    def verify_code(code):
        """
        Verifies the 2FA code.

        :param code: The 2FA code
        :return: True if the code is correct, False otherwise
        """
        return ToptCode.totp.verify(code, valid_window=1)


class ParsedUserAgent(UserAgent):
    """
    This class is a wrapper around the UserAgent class from Werkzeug.
    It parses the user agent string and provides access to the browser
    and operating system properties. It also provides a method to
    return a dictionary of the parsed user agent string.
    """

    @cached_property
    def _details(self):
        """
        Parse the user agent string and return a dictionary of the
        parsed user agent string.

        :return: A dictionary of the parsed user agent string
        """
        return user_agent_parser.Parse(self.string)

    @property
    def platform(self):
        """
        Return the operating system name.

        :return: The operating system name
        """
        return self._details['os']['family']

    @property
    def browser(self):
        """
        Return the browser name.

        :return: The browser name
        """
        return self._details['user_agent']['family']

    @property
    def version(self):
        """
        Return the browser version.

        :return: The browser version
        """
        return '.'.join(
            part
            for key in ('major', 'minor', 'patch')
            if (part := self._details['user_agent'][key]) is not None
        )

    @property
    def os_version(self):
        """
        Return the operating system version.

        :return: The operating system version
        """
        return '.'.join(
            part
            for key in ('major', 'minor', 'patch')
            if (part := self._details['os'][key]) is not None
        )


def get_os_browser_versions():
    """
    Get the User's browser and OS from the request header.

    :return: The User's browser and OS versions and time as a dictionary object
    """
    user_agent = ParsedUserAgent(request.headers.get('User-Agent'))
    return user_agent.platform, user_agent.os_version, user_agent.browser, user_agent.version, \
           datetime.now().strftime("%A, %I:%M:%S %p")
