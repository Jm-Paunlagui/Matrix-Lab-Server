import re


class InputTextValidation:
    """Validate user text input"""

    def __init__(self, user_input: str = None):
        self.user_input = user_input

    @staticmethod
    def validate_empty_fields(*args: str):
        """Checks if any of the fields are empty."""
        return all(not arg == "" or arg is None or arg == " " for arg in args)

    def validate_email(self):
        """Checks if the email is valid."""
        return bool(re.compile(r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@(["
                               r"-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])")
                    .match(self.user_input))

    def validate_password(self):
        """Checks if the password is valid."""
        return bool(re.compile(r"^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#?!@$%^&*-]).{8,}$")
                    .match(self.user_input))

    def validate_username(self):
        """Checks if the username is valid."""
        return bool(re.compile(r"^[a-zA-Z0-9_-]{5,20}$").match(self.user_input))

    def validate_text(self):
        """Checks if the text is valid."""
        return bool(re.compile(r"^[^0-9_!¡?÷¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}$").match(self.user_input))

    def validate_number(self):
        """Checks if the number is valid."""
        return bool(re.compile(r"^[0-9]+$").match(str(self.user_input)))


print(InputTextValidation("johnpaunlagui@gmail.com").validate_email())
