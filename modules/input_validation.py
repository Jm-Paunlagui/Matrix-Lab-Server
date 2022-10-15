import re


# desc: Validation for empty fields
def validate_empty_fields(*args: str):
    for arg in args:
        if arg == "" or arg is None or arg == " ":
            return False
    return True


# desc: Validation for email
def validate_email(email: str):
    if re.compile(r"[^@]+@[^@]+\.[^@]+").match(email):
        return True
    else:
        return False


# desc: Validation for password
def validate_password(password: str):
    if re.compile(r"^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$").match(password):
        return True
    else:
        return False


# desc: Validation for username
def validate_username(username: str):
    if re.compile(r"^[a-zA-Z0-9_-]{5,20}$").match(username):
        return True
    else:
        return False


# desc: Validation for text
def validate_text(text: str):
    if re.compile(r"^[^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}$").match(text):
        return True
    else:
        return False


# desc: Validation for text
def validate_number(number: int):
    if re.compile(r"^[0-9]{1,}$").match(str(number)):
        return True
    else:
        return False
