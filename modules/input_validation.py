import re


def validate_empty_fields(*args: str):
    """Checks if any of the fields are empty."""

    return all(not arg == "" or arg is None or arg == " " for arg in args)


def validate_email(email: str):
    """Checks if the email is valid."""

    return bool(re.compile(r"[^@]+@[^@]+\.[^@]+").match(email))


def validate_password(password: str):
    """Checks if the password is valid."""

    return bool(re.compile(r"^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$").match(password))


def validate_username(username: str):
    """Checks if the username is valid."""

    return bool(re.compile(r"^[a-zA-Z0-9_-]{5,20}$").match(username))


def validate_text(text: str):
    """Checks if the text is valid."""

    return bool(re.compile(r"^[^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}$").match(text))


def validate_number(number: int):
    """Checks if the number is valid."""

    return bool(re.compile(r"^[0-9]{1,}$").match(str(number)))
