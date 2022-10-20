from database_queries.user_queries import (authenticate_user,
                                           authenticated_user,
                                           check_email_exists,
                                           check_email_exists_by_username,
                                           check_password_reset_token_exists,
                                           create_user, password_reset,
                                           password_reset_link, redirect_to,
                                           remove_session, security_code)
from flask import jsonify, request
from modules.input_validation import (validate_email, validate_empty_fields,
                                      validate_password, validate_text,
                                      validate_username)


def signup():
    """User registration route handler function."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json['email']
    first_name = request.json['first_name']
    last_name = request.json['last_name']
    username = request.json['username']
    password = request.json['password']
    role = request.json['role']

    if not validate_empty_fields(email, first_name, last_name, username, password, role):
        return jsonify({"status": "error", "message": "Please fill in all the fields!"}), 400
    if not validate_email(email):
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not validate_username(username):
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    if not validate_password(password):
        return jsonify({"status": "error", "message": "Follow the password rules below!"}), 400
    if not validate_text(first_name):
        return jsonify({"status": "error", "message": "Invalid first name!"}), 400
    if not validate_text(last_name):
        return jsonify({"status": "error", "message": "Invalid last name!"}), 400
    if not validate_text(role):
        return jsonify({"status": "error", "message": "Invalid role!"}), 400
    if not create_user(email, first_name, last_name, username, password, role):
        return jsonify({"status": "error", "message": "Email already exists!"}), 409
    return jsonify({"status": "success", "message": "User account created successfully."}), 201


def authenticate():
    """User authentication route handler function and return email for identity confirmation"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    username = request.json["username"]
    password = request.json["password"]

    if not validate_empty_fields(username, password):
        return jsonify({"status": "error", "message": "Username and password are required!"}), 400
    if not validate_username(username) or not validate_password(password):
        return jsonify({"status": "error", "message": "Not a valid username or password!"}), 400
    if not authenticate_user(username, password):
        return jsonify({"status": "error", "message": "Invalid username or password!"}), 401
    identity: dict = redirect_to()
    return jsonify({"status": "success", "message": "User authenticated successfully.", "identity_one": identity[0],
                    "identity_two": identity[1], "path": identity[2]}), 200


def send_security_code():
    """Sends a security code to the email that is provided by the user. either primary email or recovery email"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]

    if not validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Email is required!"}), 400
    if not validate_email(email):
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not security_code(email):
        return jsonify({"status": "error", "message": "Security code not sent!"}), 500
    return jsonify({"status": "success", "message": "Email sent successfully."}), 200

def get_authenticated_user():
    """Gets the authenticated user by id and returns the user object."""
    user = authenticated_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized access", "path": "/"}), 401
    return jsonify({"status": "success", "message": "User retrieved successfully",
                    "user": {
                        "id": user.user_id, "email": user.email, "first_name": user.first_name,
                        "last_name": user.last_name, "username": user.username, "role": user.role, "path": redirect_to()
                    }}), 200


def signout():
    """Signs out the authenticated user by id and deletes the session."""
    if not remove_session():
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify({"status": "success", "message": "User signed out successfully"}), 200


def check_email():
    """Checks if the email address exists in the database to confirm the users email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    username = request.json["username"]

    if not validate_empty_fields(username):
        return jsonify({"status": "error", "message": "Username is required!"}), 400
    if not validate_username(username):
        return jsonify({"status": "error", "message": "Invalid username!"}), 400
    if not check_email_exists_by_username(username):
        return jsonify({"status": "error", "message": username + " does not exist!"}), 404
    return jsonify({"status": "success", "message": "Email retrieved successfully.",
                    "email": check_email_exists_by_username(username)}), 200


def forgot_password():
    """Sends a password reset link to the user's email address."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    email = request.json["email"]

    if not validate_empty_fields(email):
        return jsonify({"status": "error", "message": "Email address is required!"}), 400
    if not validate_email(email):
        return jsonify({"status": "error", "message": "Invalid email address!"}), 400
    if not check_email_exists(email):
        return jsonify({"status": "error", "message": "Email address does not exist!"}), 404
    if check_password_reset_token_exists(email):
        return jsonify({"status": "error", "message": "Password reset link already sent!"}), 409
    password_reset_link(email)
    return jsonify({"status": "success", "message": "Password reset link sent successfully, "
                                                    "Please check your email."}), 200


def reset_password(token: str):
    """Resets the user's password by inputting his/her new password."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request!"})

    password = request.json["password"]

    if not validate_empty_fields(password):
        return jsonify({"status": "error", "message": "Create a new password!"}), 400
    if not validate_password(password):
        return jsonify({"status": "error", "message": "Follow the password rules below!"}), 400
    if not password_reset(token, password):
        return jsonify({"status": "error", "message": "Link session expired!"}), 404
    return jsonify({"status": "success", "message": "Your password has been reset successfully."}), 200
