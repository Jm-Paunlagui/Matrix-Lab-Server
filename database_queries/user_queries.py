import uuid
from datetime import datetime, timedelta

import jwt
from config.configurations import app, db, mail
from flask import session, jsonify
from flask_mail import Message
from flask_session import Session
from models.user_model import User
from modules.module import (PasswordBcrypt, PayloadSignature, Timezone,
                            ToptCode, get_os_browser_versions)

# desc: Session configuration
server_session = Session(app)


def check_email_exists(email: str):
    """Check if users email exists in the database."""
    is_email: User = User.query.with_entities(User.email, User.secondary_email, User.recovery_email).filter(
        (User.email == email) | (User.secondary_email == email) | (User.recovery_email == email)).first()
    if (
            is_email is not None
            and email in (is_email.email, is_email.secondary_email, is_email.recovery_email)
    ):
        return True
    return False


def check_username_exists(username: str):
    """Checks if the username exists in the database."""
    is_username: User = User.query.filter_by(username=username).first()
    if (
            is_username is not None
            and username == is_username.username
    ):
        return True
    return False


def check_password_reset_token_exists(email: str):
    """Check if reset password token exists in the database."""
    is_token: User = User.query.with_entities(User.email, User.secondary_email, User.recovery_email,
                                              User.password_reset_token).filter(
        (User.email == email) | (User.secondary_email == email) | (User.recovery_email == email)).first()
    if is_token.password_reset_token:
        return True
    return False


# @desc: Checks if the user id exists
def check_user_id_exists(user_id: int):
    """Check if the user id exists in the database."""
    is_user_id: User = User.query.filter_by(user_id=user_id).first()
    if is_user_id.user_id:
        return True
    return False


def check_email_exists_by_username(username: str):
    """Check if the user's email exists in the database."""
    is_email: User = User.query.with_entities(User.email, User.secondary_email, User.recovery_email,
                                              User.username).filter(
        (User.username == username)).first()
    if is_email is None:
        return False
    if is_email is not None and username == is_email.username:
        payload = {
            "iss": "http://127.0.0.1:5000",
            "sub": is_email.email,
            "secondary_email": is_email.secondary_email,
            "recovery_email": is_email.recovery_email,
            "iat": Timezone("Asia/Manila").get_timezone_current_time(),
            "jti": str(uuid.uuid4())
        }
        emails = PayloadSignature(payload=payload).encode_payload()
        return emails
    return False


def create_user(email: str, first_name: str, last_name: str, username: str, password: str, role: str):
    """Creates a new user in the database."""
    if check_email_exists(email):
        return False
    hashed_password = PasswordBcrypt(password=password).password_hasher()
    new_user = User(email=email, first_name=first_name, last_name=last_name, username=username,
                    password=hashed_password, role=role)
    db.session.add(new_user)
    db.session.commit()
    return True


def delete_user(user_id: int):
    """Deletes the user's account by flagging the user's account as deleted."""
    if check_user_id_exists(user_id):
        flag_delete_user: User = User.query.filter_by(user_id=user_id).first()
        flag_delete_user.flag_deleted = True
        db.session.commit()
        return True
    return False


def delete_user_permanently(user_id: int):
    """Deletes the user's account permanently."""
    if check_user_id_exists(user_id):
        permanently_delete_user: User = User.query.filter_by(
            user_id=user_id).first()
        db.session.delete(permanently_delete_user)
        db.session.commit()
        return True
    return False


def list_flag_deleted_users():
    """Lists all the users that have been flagged as deleted."""
    flag_deleted_users: User = User.query.filter_by(flag_deleted=True).all()
    return flag_deleted_users


def restore_user(user_id: int):
    """Restores the user's account by unflagging the user's account as deleted."""
    if check_user_id_exists(user_id):
        restore_user_id: User = User.query.filter_by(user_id=user_id).first()
        restore_user_id.flag_deleted = False
        db.session.commit()
        return True
    return False


def authenticate_user(username: str, password: str):
    """
    Authenticates the user's credentials by checking if the username and password exists in the database
    and if the user's account is not flagged as deleted.
    """
    is_user: User = User.query.filter_by(username=username).first()
    if is_user is None:
        return False
    if not PasswordBcrypt(password=password).password_hash_check(is_user.password) or is_user.flag_deleted:
        return False
    session['user_id'] = is_user.user_id
    return True


def send_tfa(email: str):
    """Sends a security code to the email that is provided by the user. either primary email or recovery email"""
    # Check if the email is primary or recovery
    is_email: User = User.query.with_entities(User.email, User.secondary_email, User.recovery_email,
                                              User.username).filter(
        (User.email == email) | (User.secondary_email == email) | (User.recovery_email == email)).first()

    if email in (is_email.email, is_email.secondary_email, is_email.recovery_email):
        # Generate a link for removing the user's email if not recognized by the user using jwt
        payload = {
            "iss": "http://127.0.0.1:5000",
            "sub": email,
            "username": is_email[3],
            "iat": Timezone("Asia/Manila").get_timezone_current_time(),
            "exp": datetime.timestamp(Timezone("Asia/Manila").get_timezone_current_time() + timedelta(hours=24)),
            "jti": str(uuid.uuid4())
        }
        link = PayloadSignature(payload=payload).encode_payload()

        username = is_email[3]
        source = get_os_browser_versions()

        totp = ToptCode.topt_code()

        # Send the security code to the email
        msg = Message('Security Code - Matrix Lab',
                      sender="service.matrix.ai@gmail.com", recipients=[email])

        msg.html = f""" <!doctype html><html lang="en-US"><head> <meta content="text/html; charset=utf-8" 
        http-equiv="Content-Type"/></head><body marginheight="0" topmargin="0" marginwidth="0" style="margin: 0px; 
        background-color: #f2f3f8;" leftmargin="0"> <table cellspacing="0" border="0" cellpadding="0" width="100%" 
        bgcolor="#f2f3f8" style="@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@100;200;300;400
        ;500;600;700;800;900&display=swap');font-family: 'Montserrat', sans-serif;"> <tr> <td> <table 
        style="background-color: #f2f3f8; max-width:670px; margin:0 auto; padding: auto;" width="100%" border="0" 
        align="center" cellpadding="0" cellspacing="0"> <tr> <td style="height:30px;">&nbsp;</td></tr><tr> <td 
        style="text-align:center;"> <a href="https://rakeshmandal.com" title="logo" target="_blank"> <img width="60" 
        src="https://s.gravatar.com/avatar/e7315fe46c4a8a032656dae5d3952bad?s=80" title="logo" alt="logo"> </a> 
        </td></tr><tr> <td style="height:20px;">&nbsp;</td></tr><tr> <td> <table width="87%" border="0" align="center" 
        cellpadding="0" cellspacing="0" style="max-width:670px;background:#fff; border-radius:3px; 
        text-align:center;-webkit-box-shadow:0 6px 18px 0 rgba(0,0,0,.06);-moz-box-shadow:0 6px 18px 0 rgba(0,0,0,
        .06);box-shadow:0 6px 18px 0 rgba(0,0,0,.06);"> <tr> <td style="padding:35px;"> <h1 
        style="color:#5d6068;font-weight:700;text-align:left">Security code</h1> <p style="color:#878a92;margin:.4em 0 
        2.1875em;font-size:16px;line-height:1.625; text-align: justify;">Please use the security code for the Matrix 
        account {username}.</p><h2 style="color:#5d6068;font-weight:600;text-align:left">Security code: <span 
        style="color:#878a92;font-weight:400;">{totp}</span></h2><p style="color:#878a92;margin: 2.1875em 0 
        .4em;font-size:16px;line-height:1.625; text-align: justify;">For security, this request was received from a <b>
        {source[0]} {source[1]}</b> device using <b>{source[2]} {source[3]}</b> on <b>{source[4]}</b>.</p><p 
        style="color:#878a92;margin: .4em 0 2.1875em;font-size:16px;line-height:1.625; text-align: justify;">If you did 
        not recognize this email to your {username}'s email address, you can 
        <a href="{"http://localhost:3000/remove-email-from-account/" + link}" 
        style="color:#44578b;text-decoration:none;font-weight:bold;">click here</a> to remove the email address from 
        that account.</p><p style="color:#878a92;margin:1.1875em 0 .4em;font-size:16px;line-height:1.625;text-align: 
        left;">Thanks, <br>The Matrix Lab team. </p></td></tr></table> </td><tr> <td 
        style="height:20px;">&nbsp;</td></tr><tr> <td style="text-align:center;"> <p style="font-size:14px; color:rgba(
        124, 144, 163, 0.741); line-height:18px; margin:0 0 0;">Group 14 - Matrix Lab <br>Blk 01 Lot 18 Lazaro 3 Brgy. 3 
        Calamba City, Laguna <br>4027 Philippines</p></td></tr><tr> <td style="height:20px;">&nbsp;</td></tr></table> 
        </td></tr></table></body></html> """
        mail.send(msg)
        return True
    return False


def verify_tfa(code: str):
    """Verifies the security code that is provided by the user"""
    topt = ToptCode.verify_code(code=code)
    if topt:
        return True
    return False


def authenticated_user():
    """Checks if the user is authenticated and if the user's account is not flagged as deleted."""
    user_id: int = session.get('user_id')
    if user_id is None:
        return False
    user_data: User = User.query.filter_by(user_id=user_id).first()
    payload = {
        "sub": "user",
        "token": "true", "id": user_data.user_id,
        "email": user_data.email, "secondary_email": user_data.secondary_email,
        "recovery_email": user_data.recovery_email, "first_name": user_data.first_name,
        "last_name": user_data.last_name, "username": user_data.username, "role": user_data.role, "path": redirect_to(),
        "iat": Timezone("Asia/Manila").get_timezone_current_time(),
        "exp": datetime.timestamp(Timezone("Asia/Manila").get_timezone_current_time() + timedelta(days=24)),
        "jti": str(uuid.uuid4())
    }
    user_data_token = PayloadSignature(payload=payload).encode_payload()
    return user_data_token


def password_reset_link(email: str):
    """
    Sends the password reset link to the user's email and stores the token in the database that expires in
    24 hours.
    """
    if not check_email_exists(email):
        return False
    first_name: User = User.query.with_entities(User.first_name).filter(
        (User.email == email) | (User.secondary_email == email) | (User.recovery_email == email)).first().first_name
    payload = {
        "iss": "http://127.0.0.1:5000",
        "sub": email,
        "iat": Timezone("Asia/Manila").get_timezone_current_time(),
        "exp": datetime.timestamp(Timezone("Asia/Manila").get_timezone_current_time() + timedelta(hours=24)),
        "jti": str(uuid.uuid4())
    }
    password_reset_token = PayloadSignature(payload=payload).encode_payload()
    source = get_os_browser_versions()
    User.query.filter((User.email == email) | (User.secondary_email == email) | (User.recovery_email == email)).update(
        {"password_reset_token": password_reset_token})
    db.session.commit()
    msg = Message('Password Reset Link - Matrix Lab',
                  sender="service.matrix.ai@gmail.com", recipients=[email])
    msg.html = f""" <!doctype html><html lang="en-US"><head> <meta content="text/html; charset=utf-8"
    http-equiv="Content-Type"/></head><body marginheight="0" topmargin="0" marginwidth="0" style="margin: 0px;
    background-color: #f2f3f8;" leftmargin="0"> <table cellspacing="0" border="0" cellpadding="0" width="100%"
    bgcolor="#f2f3f8" style="@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@100;200;300;400
    ;500;600;700;800;900&display=swap');font-family: 'Montserrat', sans-serif;"> <tr> <td> <table
    style="background-color: #f2f3f8; max-width:670px; margin:0 auto; padding: auto;" width="100%" border="0"
    align="center" cellpadding="0" cellspacing="0"> <tr> <td style="height:30px;">&nbsp;</td></tr><tr> <td
    style="text-align:center;"> <a href="" title="logo" target="_blank"> <img width="60"
    src="https://s.gravatar.com/avatar/e7315fe46c4a8a032656dae5d3952bad?s=80" title="logo" alt="logo"> </a>
    </td></tr><tr> <td style="height:20px;">&nbsp;</td></tr><tr> <td> <table width="87%" border="0" align="center"
    cellpadding="0" cellspacing="0" style="max-width:670px;background:#fff; border-radius:3px;
    text-align:center;-webkit-box-shadow:0 6px 18px 0 rgba(0,0,0,.06);-moz-box-shadow:0 6px 18px 0 rgba(0,0,0,
    .06);box-shadow:0 6px 18px 0 rgba(0,0,0,.06);"> <tr> <td style="padding:35px;"> <h1
    style="color:#5d6068;font-weight:700;text-align:left">Hi {first_name},</h1> <p style="color:#878a92;margin:.4em 0
    2.1875em;font-size:16px;line-height:1.625; text-align: justify;">You recently requested to reset your password
    for your Matrix account. Use the button below to reset it. <strong>This password reset is only valid for the next
    24 hours.</strong></p><a href="{"http://localhost:3000/reset-password/" + password_reset_token}"
    style="background:#22bc66;text-decoration:none !important; font-weight:500; color:#fff;text-transform:uppercase;
    font-size:14px;padding:12px 24px;display:block;border-radius:5px;box-shadow:0 2px 3px rgba(0,0,0,.16);">Reset
    Password</a> <p style="color:#878a92;margin: 2.1875em 0 .4em;font-size:16px;line-height:1.625; text-align:
    justify;">For security, this request was received from a <b>{source[0]} {source[1]}</b> device using <b>
    {source[2]} {source[3]}</b> on <b>{source[4]}</b>.</p><p style="color:#878a92;margin: .4em 0
    2.1875em;font-size:16px;line-height:1.625; text-align: justify;">If you did not request a password reset,
    please ignore this email or contact technical support by email: <b> <a
    style="text-decoration:none;color:#878a92;"
    href="mailto:paunlagui.cs.jm@gmail.com">paunlagui.cs.jm@gmail.com</a></p><p style="color:#878a92;margin:1.1875em
    0 .4em;font-size:16px;line-height:1.625;text-align: left;">Thanks, <br>The Matrix Lab team </p><hr
    style="margin-top: 12px; margin-bottom: 12px;"> <p style="color:#878a92;margin:.4em 0
    1.1875em;font-size:13px;line-height:1.625; text-align: left;">If you&#39;re having trouble with the button above,
    copy and paste the URL below into your web browser.</p><p style="color:#878a92;margin:.4em 0
    1.1875em;font-size:13px;line-height:1.625; text-align: left;">
    {"http://localhost:3000/reset-password/" + password_reset_token}</p></td></tr></table> </td><tr> <td
    style="height:20px;">&nbsp;</td></tr><tr> <td style="text-align:center;"> <p style="font-size:14px; color:rgba(
    124, 144, 163, 0.741); line-height:18px; margin:0 0 0;">Group 14 - Matrix Lab <br>Blk 01 Lot 18 Lazaro 3 Brgy. 3
    Calamba City, Laguna <br>4027 Philippines</p></td></tr><tr> <td style="height:20px;">&nbsp;</td></tr></table>
    </td></tr></table></body></html> """
    mail.send(msg)
    return True


def password_reset(password_reset_token: str, password: str):
    """
    Resets the password of the user with the given password reset token. Returns True if successful, False
    otherwise.
    """
    try:
        email: dict = PayloadSignature(
            encoded=password_reset_token).decode_payload()
        hashed_password: str = PasswordBcrypt(
            password=password).password_hasher()
        intoken: User = User.query.filter(
            (User.email == email["sub"]) | (User.secondary_email == email["sub"]) | (User.recovery_email == email["sub"])
        ).first()
        email_name = intoken.first_name
        if intoken.password_reset_token == password_reset_token:
            intoken.password = hashed_password
            intoken.password_reset_token = None
            db.session.commit()
            source = get_os_browser_versions()
            msg = Message("Password Reset Successful",
                          sender="service.matrix.ai@gmail.com", recipients=[email["sub"]])
            msg.html = f""" <!doctype html><html lang="en-US"><head> <meta content="text/html; charset=utf-8" 
            http-equiv="Content-Type"/></head><body marginheight="0" topmargin="0" marginwidth="0" style="margin: 
            0px; background-color: #f2f3f8;" leftmargin="0"> <table cellspacing="0" border="0" cellpadding="0" 
            width="100%" bgcolor="#f2f3f8" style="@import url(
            'https://fonts.googleapis.com/css2?family=Montserrat:wght@100;200;300;400;500;600;700;800;900&display
            =swap');font-family: 'Montserrat', sans-serif;"> <tr> <td> <table style="background-color: #f2f3f8; 
            max-width:670px; margin:0 auto; padding: auto;" width="100%" border="0" align="center" cellpadding="0" 
            cellspacing="0"> <tr> <td style="height:30px;">&nbsp;</td></tr><tr> <td style="text-align:center;"> <a 
            href="https://rakeshmandal.com" title="logo" target="_blank"> <img width="60" 
            src="https://s.gravatar.com/avatar/e7315fe46c4a8a032656dae5d3952bad?s=80" title="logo" alt="logo"> </a> 
            </td></tr><tr> <td style="height:20px;">&nbsp;</td></tr><tr> <td> <table width="87%" border="0" 
            align="center" cellpadding="0" cellspacing="0" style="max-width:670px;background:#fff; border-radius:3px; 
            text-align:center;-webkit-box-shadow:0 6px 18px 0 rgba(0,0,0,.06);-moz-box-shadow:0 6px 18px 0 rgba(0,0,
            0,.06);box-shadow:0 6px 18px 0 rgba(0,0,0,.06);"> <tr> <td style="padding:35px;"> <h1 
            style="color:#5d6068;font-weight:700;text-align:left">Hi {email_name},</h1> <p 
            style="color:#878a92;margin:.4em 0 2.1875em;font-size:16px;line-height:1.625; text-align: justify;">Your 
            password has been changed successfully.</p><p style="color:#878a92;margin: 2.1875em 0 
            .4em;font-size:16px;line-height:1.625; text-align: justify;">For security, this request was received from 
            a <b>{source[0]} {source[1]}</b> device using <b>{source[2]} {source[3]}</b> on <b>{source[4]}</b>.</p><p 
            style="color:#878a92;margin: .4em 0 2.1875em;font-size:16px;line-height:1.625; text-align: justify;">If 
            you did not change your password on this time period, please contact us immediately by email: <b><a 
            style="text-decoration:none;color:#878a92;" 
            href="mailto:paunlagui.cs.jm@gmail.com">paunlagui.cs.jm@gmail.com</a></b>.</p><p 
            style="color:#878a92;margin:1.1875em 0 .4em;font-size:16px;line-height:1.625;text-align: left;">Thanks, 
            <br>The Matrix Lab team. </p></td></tr></table> </td><tr> <td style="height:20px;">&nbsp;</td></tr><tr> 
            <td style="text-align:center;"> <p style="font-size:14px; color:rgba(124, 144, 163, 
            0.741); line-height:18px; margin:0 0 0;">Group 14 - Matrix Lab <br>Blk 01 Lot 18 Lazaro 3 Brgy. 3 Calamba 
            City, Laguna <br>4027 Philippines</p></td></tr><tr> <td style="height:20px;">&nbsp;</td></tr></table> 
            </td></tr></table></body></html> """
            mail.send(msg)
            return True
        return False
    except jwt.exceptions.InvalidTokenError:
        token: User = User.query.filter_by(
            password_reset_token=password_reset_token).first()
        token.password_reset_token = None
        db.session.commit()
        return False


def has_emails():
    """Gets the email and recovery email of the user based on user session."""
    user_id: int = session.get('user_id')

    if user_id is None:
        return False

    user_email: User = User.query.with_entities(User.email, User.secondary_email, User.recovery_email) \
        .filter_by(user_id=user_id).first()

    user_emails = {
        "iss": "http://127.0.0.1:5000",
        "sub": "has_emails",
        "id1": user_email.email,
        "id2": user_email.secondary_email,
        "id3": user_email.recovery_email,
        "iat": Timezone("Asia/Manila").get_timezone_current_time(),
        "jti": str(uuid.uuid4())
    }
    return PayloadSignature(payload=user_emails).encode_payload()


def redirect_to():
    """Redirects the user to the appropriate page based on the user role."""
    user_id: int = session.get('user_id')
    user_role: User = User.query.filter_by(user_id=user_id).first()
    match user_role.role:
        case 'admin':
            return "/admin/dashboard"
        case 'user':
            return "/user/dashboard"
    return "/"


def remove_session():
    """Removes the user's session if the user logs out."""
    user_id: int = session.get('user_id')
    if user_id is not None:
        session.pop('user_id', None)
        session.clear()
        return True
    return False


def verify_remove_token(token: str):
    """Verifies the token for the user to remove their account."""
    try:
        user_info: dict = PayloadSignature(encoded=token).decode_payload()
        return user_info
    except jwt.exceptions.InvalidTokenError:
        return False


def verify_authenticated_token(token: str):
    """Verifies the token for the user to access the dashboard."""
    try:
        user_info: dict = PayloadSignature(encoded=token).decode_payload()
        return user_info
    except jwt.exceptions.InvalidTokenError:
        return False


def remove_email(option: str, email: str, username: str):
    """Removes the email address if the user does not recognize the email address"""
    remove: User = User.query.with_entities(User.email, User.secondary_email, User.recovery_email,
                                            User.username).filter(
        User.username == username).first()
    if option == "no" and remove is not None:
        if remove.email == email:
            return False
        if email in (remove.secondary_email, remove.recovery_email) and username == remove.username:
            type_of_email = "secondary_email" if remove.secondary_email == email else "recovery_email"
            User.query.filter_by(username=username).update(
                {type_of_email: None})
            source = get_os_browser_versions()
            msg = Message(subject="Email Removed",
                          sender="service.matrix.ai@gmail.com", recipients=[remove.email])
            msg.html = f"""<!DOCTYPE html><html lang="en-US"><head><meta content="text/html; charset=utf-8" 
                http-equiv="Content-Type"></head><body marginheight="0" topmargin="0" marginwidth="0" 
                style="margin:0;background-color:#f2f3f8" leftmargin="0"><table cellspacing="0" border="0" 
                cellpadding="0" width="100%" bgcolor="#f2f3f8" style="@import url(
                'https://fonts.googleapis.com/css2?family=Montserrat:wght@100;200;300;400;500;600;700;800;900&display
                =swap');font-family:Montserrat,sans-serif"><tr><td><table 
                style="background-color:#f2f3f8;max-width:670px;margin:0 auto;padding:auto" width="100%" border="0" 
                align="center" cellpadding="0" cellspacing="0"><tr><td style="height:30px">&nbsp;</td></tr><tr><td 
                style="text-align:center"><a href="https://rakeshmandal.com" title="logo" target="_blank"><img 
                width="60" src="https://s.gravatar.com/avatar/e7315fe46c4a8a032656dae5d3952bad?s=80" title="logo" 
                alt="logo"></a></td></tr><tr><td style="height:20px">&nbsp;</td></tr><tr><td><table width="87%" 
                border="0" align="center" cellpadding="0" cellspacing="0" 
                style="max-width:670px;background:#fff;border-radius:3px;text-align:center;-webkit-box-shadow:0 6px 
                18px 0 rgba(0,0,0,.06);-moz-box-shadow:0 6px 18px 0 rgba(0,0,0,.06);box-shadow:0 6px 18px 0 rgba(0,0,
                0,.06)"><tr><td style="padding:35px"><h1 style="color:#5d6068;font-weight:700;text-align:left">Email 
                removed from {username}</h1><p style="color:#878a92;margin:.4em 0 
                2.1875em;font-size:16px;line-height:1.625;text-align:justify">The email address <b><a 
                style="text-decoration:none;color:#878a92">{email}</a></b> has been removed from your Matrix account 
                {username}.</p><p style="color:#878a92;margin:2.1875em 0 
                .4em;font-size:16px;line-height:1.625;text-align:justify">For security, this request was received 
                from a <b>{source[0]} {source[1]}</b> device using <b>{source[2]} {source[3]}</b> on <b>
                {source[4]}</b>.</p><p style="color:#878a92;margin:.4em 0 
                2.1875em;font-size:16px;line-height:1.625;text-align:justify">If this was not you, please contact 
                technical support by email:<b><a style="text-decoration:none;color:#878a92" 
                href="mailto:paunlagui.cs.jm@gmail.com">paunlagui.cs.jm@gmail.com</a></b></p><p 
                style="color:#878a92;margin:1.1875em 0 .4em;font-size:16px;line-height:1.625;text-align:left">Thanks,
                <br>The Matrix Lab team.</p></td></tr></table></td></tr><tr><td 
                style="height:20px">&nbsp;</td></tr><tr><td style="text-align:center"><p 
                style="font-size:14px;color:rgba(124,144,163,.741);line-height:18px;margin:0 0 0">Group 14 - Matrix 
                Lab<br>Blk 01 Lot 18 Lazaro 3 Brgy. 3 Calamba City, Laguna<br>4027 Philippines</p></td></tr><tr><td 
                style="height:20px">&nbsp;</td></tr></table></td></tr></table></body></html> """
            mail.send(msg)
            db.session.commit()
            return True
        return False
    return False


def update_password(old_password: str, new_password: str):
    """Updates the password of the user"""
    user_id: int = session.get("user_id")
    user: User = User.query.with_entities(User.password) \
        .filter(User.user_id == user_id).first()
    if user_id is not None and user is not None and \
            PasswordBcrypt(password=old_password).password_hash_check(user.password):
        User.query.filter_by(user_id=user_id).update({
            User.password: PasswordBcrypt(
                password=new_password).password_hasher()
        })
        db.session.commit()
        return True
    return False


def update_personal_info(email: str, first_name: str, last_name: str):
    """Updates the personal information of the user"""
    user_id: int = session.get("user_id")
    user: User = User.query.with_entities(User.email, User.first_name, User.last_name) \
        .filter(User.user_id == user_id).first()
    if user_id is None and user is None:
        return jsonify({"status": "error", "message": "User not found"}), 404
    if user.email != email and check_email_exists(email):
        return jsonify({"status": "warn", "message": "Email already exists"}), 409
    User.query.filter_by(user_id=user_id).update({
        User.email: email,
        User.first_name: first_name,
        User.last_name: last_name
    })
    db.session.commit()
    return jsonify({"status": "success",
                    "message": "Your personal information has been updated successfully.",
                    "token": authenticated_user()}), 200


def update_security_info(secondary_email: str, recovery_email: str):
    """Updates the security information of the user"""
    user_id: int = session.get("user_id")
    user: User = User.query.with_entities(User.secondary_email, User.recovery_email) \
        .filter(User.user_id == user_id).first()
    if user_id is None and user is None:
        return jsonify({"status": "error", "message": "User not found"}), 404
    if user.secondary_email != secondary_email and check_email_exists(secondary_email):
        return jsonify({"status": "warn", "message": "Email already exists"}), 409
    if user.recovery_email != recovery_email and check_email_exists(recovery_email):
        return jsonify({"status": "warn", "message": "Email already exists"}), 409
    User.query.filter_by(user_id=user_id).update({
        User.secondary_email: secondary_email,
        User.recovery_email: recovery_email
    })
    db.session.commit()
    return jsonify({"status": "success",
                    "message": "Your security information has been updated successfully.",
                    "token": authenticated_user()}), 200


def update_username(username: str):
    """Updates the username of the user"""
    user_id: int = session.get("user_id")
    user: User = User.query.with_entities(
        User.username).filter(User.user_id == user_id).first()
    if user_id is None and user is None:
        return jsonify({"status": "error", "message": "User not found"}), 404
    if user.username != username and check_username_exists(username):
        return jsonify({"status": "warn", "message": "Username already exists"}), 409
    User.query.filter_by(user_id=user_id).update({
        User.username: username
    })
    db.session.commit()
    return jsonify({"status": "success",
                    "message": "Your username has been updated successfully.",
                    "token": authenticated_user()}), 200
