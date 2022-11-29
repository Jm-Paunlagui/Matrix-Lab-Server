from flask import jsonify, request

from database_queries.user_queries import (authenticate_user,
                                           authenticated_user,
                                           check_email_exists,
                                           check_email_exists_by_username,
                                           check_password_reset_token_exists,
                                           check_username_exists, create_user,
                                           has_emails, password_reset,
                                           password_reset_link, redirect_to,
                                           remove_email, remove_session,
                                           send_tfa, update_password,
                                           update_personal_info,
                                           update_security_info,
                                           update_username,
                                           verify_authenticated_token,
                                           verify_remove_token,
                                           verify_reset_token, verify_tfa, create_user_auto_generated_password,
                                           lock_user_account, unlock_user_account)
from modules.module import InputTextValidation


def authenticate():
    """User authentication route handler function and return email for identity confirmation"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    username: str = request.json["username"]
    password: str = request.json["password"]

    if not InputTextValidation().validate_empty_fields(username, password):
        return jsonify({"status": "error", "message": "All fields are required!"}), 400
    if not InputTextValidation(username).validate_username() or not InputTextValidation(password).validate_password():
        return jsonify({"status": "error", "message": "Not a valid username or password!"}), 400
    if not authenticate_user(username, password):
        return jsonify({"status": "error", "message": "Invalid username or password!"}), 401
    return jsonify({"status": "success", "message": "User authenticated successfully.", "emails": has_emails()}), 200


def check_email():
    """Checks if the email address exists in the database to confirm the users email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    username = request.json["username"]

    if not InputTextValidation().validate_empty_fields(username):
        return jsonify({"status": "error", "message": "Username is required!"}), 400
    if not InputTextValidation(username).validate_username():
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    if not check_email_exists_by_username(username):
        return jsonify({"status": "warn", "message": username + " does not exist!"}), 404
    return jsonify({"status": "success", "message": "Email retrieved successfully.",
                    "emails": check_email_exists_by_username(username)}), 200


def send_security_code():
    """Sends a security code to the email that is provided by the user. either primary email or recovery email"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]

    if not InputTextValidation().validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Choose an email!"}), 400
    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not send_tfa(email):
        return jsonify({"status": "error", "message": "Security code not sent!"}), 500
    return jsonify({"status": "success", "message": "Security code sent successfully."}), 200


def forgot_password():
    """Sends a password reset link to the user's email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]
    confirm_email = request.json["confirm_email"]

    if not InputTextValidation().validate_empty_fields(email, confirm_email):
        return jsonify({"status": "error", "message": "Email address is required!"}), 400
    if not InputTextValidation(email).validate_email() or not InputTextValidation(confirm_email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if email != confirm_email:
        return jsonify({"status": "error", "message": "Email addresses do not match!"}), 400
    if not check_email_exists(email):
        return jsonify({"status": "warn", "message": "Email address does not exist!"}), 404
    if check_password_reset_token_exists(email):
        return jsonify({"status": "warn", "message": "Password reset link already sent!"}), 409
    password_reset_link(email)
    return jsonify({"status": "success", "message": "Password reset link sent successfully."}), 200


def get_authenticated_user():
    """Gets the authenticated user by id and returns the user object."""
    token: str = request.headers["Authorization"]
    if not token:
        return jsonify({"status": "error", "message": "Invalid request!"}), 400

    verified_token: dict = verify_authenticated_token(token)
    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401
    return jsonify({"status": "success", "message": "User retrieved successfully.",
                    "user": verified_token}), 200


def remove_email_from_account():
    """Removes the email address from the user's account."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    option = request.json["option"]
    email = request.json["email"]
    username = request.json["username"]

    if not InputTextValidation().validate_empty_fields(option, email, username):
        return jsonify({"status": "error", "message": "Field required!"}), 400
    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not InputTextValidation(username).validate_username():
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    if not check_email_exists(email):
        return jsonify({"status": "warn", "message": "Email address does not exist!"}), 404
    if not check_username_exists(username):
        return jsonify({"status": "warn", "message": "Username does not exist!"}), 404
    if option == "yes":
        return jsonify({"status": "success", "message": "No changes made to your account."}), 200
    if not remove_email(option, email, username):
        return jsonify({"status": "error", "message": "You can't remove the email address that is associated with "
                                                      "your account."}), 409
    return jsonify({"status": "success", "message": "Email address removed successfully."}), 200


def reset_password(token: str):
    """Resets the user's password by inputting his/her new password."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    password = request.json["password"]

    if not InputTextValidation().validate_empty_fields(password):
        return jsonify({"status": "error", "message": "Create a new password!"}), 400
    if not InputTextValidation(password).validate_password():
        return jsonify({"status": "error", "message": "Follow the password rules below!"}), 400
    if not password_reset(token, password):
        return jsonify({"status": "error", "message": "Link session expired!"}), 404
    return jsonify({"status": "success", "message": "Your password has been reset successfully."}), 200


def signout():
    """Signs out the authenticated user by id and deletes the session."""
    if not remove_session():
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify({"status": "success", "message": "User signed out successfully"}), 200


def signup():
    """User registration route handler function."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json['email']
    full_name = request.json['full_name']
    username = request.json['username']
    password = request.json['password']
    role = request.json['role']

    if not InputTextValidation().validate_empty_fields(email, full_name, username, password, role):
        return jsonify({"status": "error", "message": "Please fill in all the fields!"}), 400
    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not InputTextValidation(username).validate_username():
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    if not InputTextValidation(password).validate_password():
        return jsonify({"status": "error", "message": "Follow the password rules below!"}), 400
    if not InputTextValidation(full_name).validate_text():
        return jsonify({"status": "error", "message": "Invalid first name!"}), 400
    if not InputTextValidation(role).validate_text():
        return jsonify({"status": "error", "message": "Invalid role!"}), 400
    if not create_user(email, full_name, username, password, role):
        return jsonify({"status": "warn", "message": "Email already exists!"}), 409
    return jsonify({"status": "success", "message": "User account created successfully."}), 201


def one_click_create(user_id: int):
    """Creates a new user account by one-click registration."""
    if not create_user_auto_generated_password(user_id):
        return jsonify({"status": "error", "message": "User account already exists!"}), 409
    return jsonify({"status": "success", "message": "User account created successfully."}), 201


def lock_user_account_by_id(user_id: int):
    """Locks the user account by id."""
    if not lock_user_account(user_id):
        return jsonify({"status": "error", "message": "Account already locked!"}), 409
    return jsonify({"status": "success", "message": "User account locked successfully."}), 200


def unlock_user_account_by_id(user_id: int):
    """Unlocks the user account by id."""
    if not unlock_user_account(user_id):
        return jsonify({"status": "error", "message": "Account already unlocked!"}), 409
    return jsonify({"status": "success", "message": "User account unlocked successfully."}), 200


def update_user_password():
    """Updates the user's password by inputting his/her current password and new password."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    current_password = request.json["old_password"]
    new_password = request.json["new_password"]

    if not InputTextValidation().validate_empty_fields(current_password, new_password):
        return jsonify({"status": "error", "message": "Field required!"}), 400
    if not InputTextValidation(new_password).validate_password():
        return jsonify({"status": "error", "message": "Follow the password rules below!"}), 400
    if not update_password(current_password, new_password):
        return jsonify({"status": "error", "message": "Current password is incorrect!"}), 401
    return jsonify({"status": "success",
                    "message": "Your password has been updated successfully."}), 200


def update_user_personal_info():
    """Updates the user's personal information by inputting his/her new information."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]
    full_name = request.json["full_name"]

    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not InputTextValidation(full_name).validate_text():
        return jsonify({"status": "error", "message": "Invalid first name!"}), 400
    return update_personal_info(email, full_name, )


def update_user_security_info():
    """Updates the user's security emails to be an extra option to send the 2FA codes"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    secondary_email = request.json["secondary_email"]
    recovery_email = request.json["recovery_email"]

    if not InputTextValidation(secondary_email or recovery_email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if secondary_email == recovery_email:
        return jsonify({"status": "error", "message": "Emails cannot be the same!"}), 400
    return update_security_info(secondary_email, recovery_email)


def update_user_username():
    """Updates the user's username by making sure the username is not taken."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    username = request.json["username"]

    if not InputTextValidation().validate_empty_fields(username):
        return jsonify({"status": "error", "message": "Field required!"}), 400
    if not InputTextValidation(username).validate_username():
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    return update_username(username)


def verify_security_code():
    """Verifies the security code that was sent to the user's email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    code = request.json["code"]

    if not InputTextValidation().validate_empty_fields(code):
        return jsonify({"status": "error", "message": "2FA Code are required!"}), 400
    if not InputTextValidation(code).validate_number() and len(code) != 7:
        return jsonify({"status": "error", "message": "Invalid 2FA Code!"}), 400
    if not verify_tfa(code):
        return jsonify({"status": "error", "message": "Invalid security code!"}), 401
    return jsonify({"status": "success",
                    "message": "Security code verified successfully",
                    "path": redirect_to(),
                    "token": authenticated_user(),
                    }), 200


def verify_remove_account_token(token: str):
    """Verifies the remove account token that was sent to the user's email address."""
    user_data = verify_remove_token(token)
    if not user_data:
        return jsonify({"status": "error", "message": "Invalid token!"}), 498
    return jsonify({"status": "success", "message": "Token verified successfully",
                    "user_data": {"email": user_data["sub"], "username": user_data["username"]}}), 200


def verify_reset_password_token(token: str):
    """Verifies the reset password token that was sent to the user's email address."""
    user_data = verify_reset_token(token)
    if not user_data:
        return jsonify({"status": "error", "message": "Invalid token!"}), 498
    return jsonify({"status": "success", "message": "Token verified successfully",
                    "user_data": {"email": user_data["sub"]}}), 200
