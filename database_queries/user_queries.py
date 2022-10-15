import uuid
from datetime import datetime, timedelta

import jwt
from config.configurations import app, db, mail, private_key, public_key
from flask import session
from flask_mail import Message
from flask_session import Session
from models.user_model import User
from modules.datetime_tz import timezone_current_time
from modules.password_bcrypt import password_hash_check, password_hasher
from modules.user_agent import get_os_browser_versions

# desc: Session configuration
server_session = Session(app)


# @desc: Check if users email exists
def check_email_exists(email: str):
    is_email: User = User.query.filter_by(email=email).first()
    if is_email:
        return True
    return False


# @desc: Check if reset password token exists
def check_password_reset_token_exists(email: str):
    # desc: Check if reset password token exists in user email address
    is_token: User = User.query.filter_by(email=email).first()
    if is_token.password_reset_token:
        return True
    return False


# @desc: Checks if the user id exists
def check_user_id_exists(user_id: int):
    # desc: Check if user id exists
    is_user_id: User = User.query.filter_by(user_id=user_id).first()
    if is_user_id.user_id:
        return True
    return False


# @desc: Check email if it exists by username
def check_email_exists_by_username(username: str):
    # desc: Check if email exists by username
    is_email: User = User.query.filter_by(username=username).first()
    if is_email is None:
        return False
    if is_email.username == username:
        # Hide some text in the email address and hide the domain name of the email address
        return is_email.email[:2] + '*****' + is_email.email[is_email.email.find('@'):]
    return False


# @desc: Creates a new user account
def create_user(email: str, first_name: str, last_name: str, username: str, password: str, role: str):
    # @desc: Check if the user's email exists
    if check_email_exists(email):
        return False

    # @desc: Hash the user's password before storing it in the database
    hashed_password = password_hasher(password)

    # @desc: Insert the user's data into the database
    new_user = User(email=email, first_name=first_name, last_name=last_name, username=username,
                    password=hashed_password, role=role)
    db.session.add(new_user)
    db.session.commit()
    return True


# @desc: Flags delete the user's account based on the user's id
def delete_user(user_id: int):
    # @desc: Check if the user's id exists
    if check_user_id_exists(user_id):
        # @desc: Flag delete the user's account
        flag_delete_user: User = User.query.filter_by(user_id=user_id).first()
        flag_delete_user.flag_deleted = True
        db.session.commit()
        return True
    return False


# @desc: Permanently deletes the user's account based on the user's id
def delete_user_permanently(user_id: int):
    # @desc: Check if the user's id exists
    if check_user_id_exists(user_id):
        # @desc: Delete the user's account
        permanently_delete_user: User = User.query.filter_by(
            user_id=user_id).first()
        db.session.delete(permanently_delete_user)
        db.session.commit()
        return True
    return False


# @desc: Lists the flagged deleted users
def list_flag_deleted_users():
    # @desc: List the flagged deleted users
    flag_deleted_users: User = User.query.filter_by(flag_deleted=True).all()
    return flag_deleted_users


# @desc: Restores the user's account based on the user's id
def restore_user(user_id: int):
    # @desc: Check if the user's id exists
    if check_user_id_exists(user_id):
        # @desc: Restore the user's account
        restore_user_id: User = User.query.filter_by(user_id=user_id).first()
        restore_user_id.flag_deleted = False
        db.session.commit()
        return True
    return False


# @desc: For user authentication
def authenticate_user(username: str, password: str):
    # desc: Check if the user exists
    is_user: User = User.query.filter_by(username=username).first()

    if is_user is None:
        return False

    # desc: Check if the user's password is correct
    if not password_hash_check(is_user.password, password) or is_user.flag_deleted:
        return False

    # desc: Generate a session token
    session['user_id'] = is_user.user_id
    return True


# @desc: Gets the user's id
def authenticated_user():
    # @desc: Get the user's session id
    user_id: int = session.get('user_id')

    # @desc: Check if the user's session exists
    if user_id is None:
        return False

    # @desc: Get the user's data
    user_data: User = User.query.filter_by(user_id=user_id).first()
    return user_data


# @desc: Gets the user's role from the database and redirects to the appropriate page
def redirect_to():
    # @desc: Get the user's session
    user_id = session.get('user_id')

    # @desc: Get role from the database
    user_role: User = User.query.filter_by(user_id=user_id).first()

    # @desc: Redirect to the appropriate page
    match user_role.role:
        case 'admin':
            return '/admin/dashboard'
        case 'user':
            return '/user/dashboard'
    return "/"


# @desc: Removes the user's session
def remove_session():
    # @desc: Get the user's session
    user_id = session.get('user_id')

    # @desc: Check if the user's session exists
    if user_id is not None:
        session.pop('user_id', None)
        session.clear()
        return True
    return False


# @desc: Generates a password reset token
def password_reset_link(email: str):

    if not check_email_exists(email):
        return False

    # desc: Gets the emails first name from the database
    first_name: User = User.query.filter_by(email=email).first().first_name

    # desc: Generates a payload for the password reset token
    payload = {
        "iss": "http://127.0.0.1:5000",
        "sub": email,
        "iat": datetime.timestamp(timezone_current_time),
        "exp": datetime.timestamp(timezone_current_time + timedelta(hours=24)),
        "jti": str(uuid.uuid4())
    }

    # @desc: Generate the password reset link
    password_reset_token = jwt.encode(payload, private_key, algorithm="RS256")

    # @desc: Get the source of the request
    source = get_os_browser_versions()

    # @desc: Update the password reset token in the database
    User.query.filter_by(email=email).update(
        {"password_reset_token": password_reset_token})
    db.session.commit()

    # @desc: Send the password reset link to the user's email address
    msg = Message('Password Reset Link - Matrix Lab',
                  sender="service.matrix.ai@gmail.com", recipients=[email])

    # @desc: The email's content and format (HTML)
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

    # @desc: Send the email
    mail.send(msg)
    return True


# @desc: Reset the user's password
def password_reset(password_reset_token: str, password: str):
    try:
        # @desc: Get the user's email address from the password reset link
        email: dict = jwt.decode(password_reset_token, public_key, algorithms=[
                                 "RS256"], verify=True)

        # @desc: Hash the user's password
        hashed_password: str = password_hasher(password)

        # @desc: Check if the token is still in the database, if it is, reset the user's password, if not, return False
        intoken: User = User.query.filter_by(email=email["sub"]).first()
        email_name = intoken.first_name
        if intoken.password_reset_token == password_reset_token:
            # @desc: Update the user's password and set the password reset token to NULL
            new_password: User = User.query.filter_by(
                email=email["sub"]).first()
            new_password.password = hashed_password
            new_password.password_reset_token = ""
            db.session.commit()

            # desc: Gets the source of the request
            source = get_os_browser_versions()

            # desc: Email the user that their password has been reset successfully with a device and browser
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
        # @desc: If the password reset link has expired, tampers with the link, or the link is invalid, return False
        return False
