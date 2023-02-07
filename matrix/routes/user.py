from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

from matrix.controllers.user import authenticate_user, send_tfa, check_email_exists, send_email_verification, \
    password_reset_link, check_username_exists, remove_email, password_reset, \
    remove_session, create_user, create_user_auto_generated_password, create_all_users_auto_generated_password, \
    deactivate_user, deactivate_all_users, lock_user_account, lock_all_user_accounts, check_email_exists_by_username, \
    unlock_user_account, unlock_all_user_accounts, delete_user_account, delete_all_user_accounts, \
    restore_user_account, restore_all_user_accounts, update_password, update_personal_info, update_security_info, \
    update_username, verify_tfa, redirect_to, authenticated_user, verify_verification_code_to_unlock, \
    verify_remove_token, verify_token, send_username_to_email, verify_email_request, verify_email
from matrix.models.user import User
from matrix.module import InputTextValidation, verify_authenticated_token

user = Blueprint("user", __name__, url_prefix="/user")


@user.route("/test", methods=["GET"])
def test():
    return "Hello, World!"


@user.route("/authenticate", methods=["POST"])
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
    return authenticate_user(username, password)
    #     return jsonify({"status": "error", "message": "Invalid username or password!"}), 401
    # return jsonify({"status": "success", "message": "User authenticated successfully.", "emails": has_emails()}), 200


@user.route("/check-email", methods=["POST"])
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


@user.route("/checkpoint-2fa", methods=["POST"])
def send_security_code():
    """Sends a security code to the email that is provided by the user. either primary email or recovery email"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]

    if not InputTextValidation().validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Choose an email!"}), 400
    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    return send_tfa(email=email, type_of_tfa="default")


@user.route("/checkpoint-2fa-email", methods=["POST"])
def send_security_code_email():
    """Sends a security code to the email that is provided by the user. either primary email or recovery email"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]

    if not InputTextValidation().validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Email required!"}), 400
    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    return send_tfa(email=email, type_of_tfa="email")


@user.route("/send-verification-code", methods=["POST"])
def send_verification_code():
    """Sends a verification code to the user's email address."""
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
    return send_email_verification(email)


@user.route("/forgot-password", methods=["POST"])
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
    if not password_reset_link(email):
        return jsonify({"status": "error", "message": "Password reset link not sent!"}), 500
    return jsonify({"status": "success", "message": "Password reset link sent successfully."}), 200


@user.route("/get_user/<string:token>", methods=["GET"])
def get_authenticated_user(token: str):
    """Gets the authenticated user by id and returns the user object."""

    if token is None:
        return jsonify({"status": "error", "message": "You are not logged in."}), 440

    verified_token: dict = verify_authenticated_token(token)

    if not verified_token:
        return jsonify({"status": "error", "message": "Invalid token!"}), 401

    if bool(verified_token["id"]):
        return jsonify({"status": "success", "message": "User retrieved successfully.",
                        "user": verified_token}), 200
    remove_session(token)
    return jsonify({
        "status": "error",
        "message": "Token and User ID did not match"
    }), 401


@user.route("/remove-email-from-account", methods=["POST"])
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


@user.route("/reset-password/<string:token>", methods=["POST"])
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


@user.route("/sign-out/<string:token>", methods=["POST"])
def signout(token: str):
    """Signs out the authenticated user by id and deletes the session."""
    if not remove_session(token):
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify({"status": "success", "message": "User signed out successfully"}), 200


@user.route("/signup", methods=["POST"])
def signup():
    """User registration route handler function."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]
    full_name = request.json["full_name"]
    username = request.json["username"]
    password = request.json["password"]
    role = request.json["role"]

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


@user.route("/on-click-create/<int:user_id>", methods=["POST"])
def one_click_create(user_id: int):
    """Creates a new user account by one-click registration."""
    if not create_user_auto_generated_password(user_id):
        return jsonify({"status": "error", "message": "User account already activated!"}), 409
    return jsonify({"status": "success", "message": "User account successfully activated."}), 201


@user.route("/mass-create-all", methods=["POST"])
def one_click_create_all():
    """Creates new user accounts by one-click registration."""
    return create_all_users_auto_generated_password()


@user.route("/on-click-deactivate/<int:user_id>", methods=["POST"])
def one_click_deactivate(user_id: int):
    """Deactivates a user account by one-click deactivation."""
    if not deactivate_user(user_id):
        return jsonify({"status": "error", "message": "User account already deactivated!"}), 409
    return jsonify({"status": "success", "message": "User account deactivated successfully."}), 200


@user.route("/mass-deactivate-all", methods=["POST"])
def one_click_deactivate_all():
    """Deactivates all user accounts by one-click deactivation."""
    return deactivate_all_users()


@user.route("/lock-account/<int:user_id>", methods=["POST"])
def lock_user_account_by_id(user_id: int):
    """Locks the user account by id."""
    if not lock_user_account(user_id):
        return jsonify({"status": "error", "message": "Account already locked!"}), 409
    return jsonify({"status": "success", "message": "User account locked successfully."}), 200


@user.route("/mass-lock-account", methods=["POST"])
def lock_all_user_account():
    """Locks all user accounts."""
    return lock_all_user_accounts()


@user.route("/unlock-account/<int:user_id>", methods=["POST"])
def unlock_user_account_by_id(user_id: int):
    """Unlocks the user account by id."""
    if not unlock_user_account(user_id):
        return jsonify({"status": "error", "message": "Account already unlocked!"}), 409
    return jsonify({"status": "success", "message": "User account unlocked successfully."}), 200


@user.route("/mass-unlock-account", methods=["POST"])
def unlock_all_user_account():
    """Unlocks all user accounts."""
    return unlock_all_user_accounts()


@user.route("/delete-account/<int:user_id>", methods=["DELETE"])
def delete_user_account_by_id(user_id: int):
    """Deletes the user account by id."""
    if not delete_user_account(user_id):
        return jsonify({"status": "error", "message": "User account not found!"}), 404
    return jsonify({"status": "success", "message": "User account deleted successfully."}), 200


@user.route("/mass-delete-account", methods=["PUT"])
def delete_all_user_account():
    """Deletes all user accounts."""
    return delete_all_user_accounts()


@user.route("/restore-account/<int:user_id>", methods=["POST"])
def restore_user_account_by_id(user_id: int):
    """Restores the user account by id."""
    if not restore_user_account(user_id):
        return jsonify({"status": "error", "message": "User account not found!"}), 404
    return jsonify({"status": "success", "message": "User account restored successfully."}), 200


@user.route("/mass-restore-account", methods=["POST"])
def restore_all_user_account():
    """Restores all user accounts."""
    return restore_all_user_accounts()


@user.route("/update-password", methods=["PUT"])
def update_user_password():
    """Updates the user's password by inputting his/her current password and new password."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    current_password = request.json["old_password"]
    new_password = request.json["new_password"]
    token = request.json["token"]

    if not InputTextValidation().validate_empty_fields(current_password, new_password):
        return jsonify({"status": "error", "message": "Field required!"}), 400
    if not InputTextValidation(new_password).validate_password():
        return jsonify({"status": "error", "message": "Follow the password rules below!"}), 400
    if not update_password(current_password, new_password, token):
        return jsonify({"status": "error", "message": "Current password is incorrect!"}), 401
    return jsonify({"status": "success",
                    "message": "Your password has been updated successfully."}), 200


@user.route("/update-personal-info", methods=["PUT"])
def update_user_personal_info():
    """Updates the user's personal information by inputting his/her new information."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]
    full_name = request.json["full_name"]
    token = request.json["token"]

    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not InputTextValidation(full_name).validate_text():
        return jsonify({"status": "error", "message": "Invalid first name!"}), 400
    return update_personal_info(email, full_name, token)


@user.route("/update-security-info", methods=["PUT"])
def update_user_security_info():
    """Updates the user's security emails to be an extra option to send the 2FA codes"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    recovery_email = request.json["recovery_email"]
    token = request.json["token"]

    if not InputTextValidation(recovery_email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    return update_security_info(recovery_email, token)


@user.route("/update-username", methods=["PUT"])
def update_user_username():
    """Updates the user's username by making sure the username is not taken."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    username = request.json["username"]
    token = request.json["token"]

    if not InputTextValidation().validate_empty_fields(username):
        return jsonify({"status": "error", "message": "Field required!"}), 400
    if not InputTextValidation(username).validate_username():
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    return update_username(username, token)


@user.route("/verify-2fa", methods=["POST"])
def verify_security_code():
    """Verifies the security code that was sent to the user's email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    code = request.json["code"]
    username = request.json["username"]

    user_id = User.query.filter_by(username=username).first().user_id

    if not InputTextValidation().validate_empty_fields(code):
        return jsonify({"status": "error", "message": "2FA Code are required!"}), 400
    if not InputTextValidation(code).validate_number() and len(code) != 7:
        return jsonify({"status": "error", "message": "Invalid 2FA Code!"}), 400
    if not verify_tfa(code):
        return jsonify({"status": "error", "message": "Invalid security code!"}), 401
    return jsonify({"status": "success",
                    "message": "Security code verified successfully",
                    "path": redirect_to(user_id=user_id),
                    "token": authenticated_user(user_id=user_id),
                    }), 200


@user.route("/verify-2fa-email", methods=["POST"])
def verify_security_code_email():
    """Verifies the security code that was sent to the user's email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    code = request.json["code"]
    email = request.json["email"]

    if not InputTextValidation().validate_empty_fields(code):
        return jsonify({"status": "error", "message": "2FA Code are required!"}), 400
    if not InputTextValidation().validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Email is required!"}), 400
    if not InputTextValidation(code).validate_number() and len(code) != 7:
        return jsonify({"status": "error", "message": "Invalid 2FA Code!"}), 400
    return send_username_to_email(code, email)


@user.route("/verify-verification-code-to-unlock", methods=["POST"])
def unlock_admin_account():
    """Unlocks the admin account by inputting the admin's username and password."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    code = request.json["code"]
    email = request.json["email"]

    if not InputTextValidation().validate_empty_fields(code):
        return jsonify({"status": "error", "message": "2FA Code are required!"}), 400
    if not InputTextValidation().validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Email is required!"}), 400
    if not InputTextValidation(code).validate_number() and len(code) != 7:
        return jsonify({"status": "error", "message": "Invalid 2FA Code!"}), 400
    return verify_verification_code_to_unlock(code, email)


@user.route("/verify-remove-account-token/<string:token>", methods=["GET"])
def verify_remove_account_token(token: str):
    """Verifies the remove account token that was sent to the user's email address."""
    user_data = verify_remove_token(token)
    if not user_data:
        return jsonify({"status": "error", "message": "Invalid token!"}), 498
    return jsonify({"status": "success", "message": "Token verified successfully",
                    "user_data": {"email": user_data["sub"], "username": user_data["username"]}}), 200


@user.route("/verify-reset-password-token/<string:token>", methods=["GET"])
def verify_reset_password_token(token: str):
    """Verifies the reset password token that was sent to the user's email address."""
    user_data = verify_token(token)
    if not user_data:
        return jsonify({"status": "error", "message": "Invalid token!"}), 498
    return jsonify({"status": "success", "message": "Token verified successfully",
                    "user_data": {"email": user_data["sub"]}}), 200


@user.route("/verify-unlock-token/<string:token>", methods=["GET"])
def verify_unlock_token(token: str):
    """Verifies to unlock token that was sent to the user's email address."""
    user_data = verify_token(token)
    if not user_data:
        return jsonify({"status": "error", "message": "Invalid token!"}), 498
    return jsonify({"status": "success", "message": "Token verified successfully",
                    "user_data": {"name": user_data["sub"]}}), 200


@user.route("/verify-email", methods=["POST"])
def verify_user_email_request():
    """Verifies the user's email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]

    if not InputTextValidation().validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Email is required!"}), 400
    if not InputTextValidation(email).validate_email():
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    return verify_email_request(email)


@user.route("/verify-email/<string:token>", methods=["GET"])
def verify_user_email(token: str):
    """Verifies the user's email address."""
    return verify_email(token)
