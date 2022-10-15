import re


# desc: Validation for empty fields
def validate_empty_fields(*args: str):
    for arg in args:
        if arg == "" or arg is None or arg == " ":
            return False
    return True


# desc: Validation for email
def validate_email(email: str):
    return bool(re.compile(r"[^@]+@[^@]+\.[^@]+").match(email))


# desc: Validation for password
def validate_password(password: str):
    return bool(re.compile(r"^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$").match(password))


# desc: Validation for username
def validate_username(username: str):
    return bool(re.compile(r"^[a-zA-Z0-9_-]{5,20}$").match(username))


# desc: Validation for text
def validate_text(text: str):
    return bool(re.compile(r"^[^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}$").match(text))


# desc: Validation for text
def validate_number(number: int):
    return bool(re.compile(r"^[0-9]{1,}$").match(str(number)))

