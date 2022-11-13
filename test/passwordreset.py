import jwt

from models.user_model import User
from modules.module import PayloadSignature, PasswordBcrypt


def password_reset(password_reset_token: str, password: str):
    """
    Resets the password of the user with the given password reset token. Returns True if successful, False
    otherwise.
    """
    try:
        email: dict = PayloadSignature(
            encoded=password_reset_token).decode_payload()
        print(email)
        hashed_password: str = PasswordBcrypt(
            password=password).password_hasher()
        print(hashed_password)
        intoken: User = User.query.filter(
            (User.email == email["sub"]) | (User.secondary_email == email["sub"]) | (User.recovery_email == email["sub"])
        ).first()
        print(intoken)

        email_name = intoken.first_name
        print(email_name)
        if intoken.password_reset_token == password_reset_token:
            intoken.password = hashed_password
            intoken.password_reset_token = None
            print(intoken.password)
            print(intoken.password_reset_token)

        return False
    except jwt.exceptions.InvalidTokenError:

        return False


token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwOi8vMTI3LjAuMC4xOjUwMDAiLCJzdWIiOiJqb2hucGF1bmxhZ3VpQGdtYWlsLmNvbSIsImlhdCI6MTY2NzYxMzc1NywiZXhwIjoxNjY3NzAwMTU3LjY5NjQ0NCwianRpIjoiMTAzNDliZGUtYWYzYy00ZDI2LWJjNWMtNWZhOWIzZTk4OTk2In0.WwF1NqJikUHc4k6WeEQyxkGV67K38m65wLKsuLiZEnNFOyzw2xC5JEsOaJCEW1o9lONnRQpmx_66RCDzuetHsZmw8aVAlHL7gEGSHr2ZiweF55udF-zM352A_1R4WeMCzAJY69bbJyPeiXXpUUVmNTv9RWJFO3CiuwN_yibq5UcTguEFPfyRHWnbRdgS6Z9z0XCWbGTJ2zITykg94E2cc3YiR8kVDHfy2gnnJo5mafUlzPI0jzVTUXPXTbvNvVmYNY0xR20tuWvdp9OyCqOOoVbZ0c0AUWRIYjhEl_6pv-3zBif_i__55YNua0u__116B2FQmsrLn1GcHLnpYiLZ9g"
password_reset(token, "password")
