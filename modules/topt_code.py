import os

import pyotp


def topt_code():
    """Generates a 6 digit code for 2FA. The code expires after 5 minutes."""

    totp = pyotp.TOTP(os.getenv("SECRET_KEY_BASE32"), digits=7, interval=30)
    return totp.now()


def verify_code(code):
    """Verifies the 2FA code."""

    totp = pyotp.TOTP(os.getenv("SECRET_KEY_BASE32"), digits=7, interval=30)
    return totp.verify(code)
