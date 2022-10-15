
from config.configurations import app
from database_queries.user_queries import (authenticate_user,
                                           authenticated_user,
                                           check_email_exists,
                                           check_email_exists_by_username,
                                           check_password_reset_token_exists,
                                           create_user, password_reset,
                                           password_reset_link, redirect_to,
                                           remove_session)
from flask import jsonify, request
from modules.input_validation import (validate_email, validate_empty_fields,
                                      validate_password, validate_text,
                                      validate_username)


# @desc: User registration route
# @app.route('/signup', methods=['POST'])
def signup():
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

    # @desc: Check if the user's email exists
    if not create_user(email, first_name, last_name, username, password, role):
        return jsonify({"status": "error", "message": "Email already exists!"}), 409

    return jsonify({"status": "success", "message": "User account created successfully."}), 201


# @desc User authentication
# @app.route("/authenticate", methods=["POST"])
def authenticate():
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

    return jsonify({"status": "success", "message": "User authenticated successfully.",
                    "path": redirect_to()}), 200


# @desc: Gets the authenticated user by id
# @app.route("/get_user", methods=["GET"])
def get_authenticated_user():
    user = authenticated_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized access", "path": "/"}), 401

    return jsonify({"status": "success", "message": "User retrieved successfully",
                    "user": {"id": user.user_id, "email": user.email, "first_name": user.first_name,
                             "last_name": user.last_name, "username": user.username, "role": user.role}}
                   ), 200


# @desc: Signs out the authenticated user by id and deletes the session
# @app.route("/sign-out", methods=["GET"])
def signout():
    if not remove_session():
        return jsonify({"status": "error", "message": "Session not found"}), 404

    return jsonify({"status": "success", "message": "User signed out successfully"}), 200


# @desc: Check email by username
# @app.route("/check-email", methods=["POST"])
def check_email():
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


# @desc: Sends a password reset link to the user's email address
# @app.route("/forgot-password", methods=["POST"])
def forgot_password():
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

    # @desc: generates a new password and sends it to the user
    password_reset_link(email)

    return jsonify({"status": "success", "message": "Password reset link sent successfully, "
                                                    "Please check your email."}), 200


# @desc: Resets the password of the user based on the token sent to the user's email address
# @app.route("/reset-password/<token>", methods=["POST"])
def reset_password(token: str):

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
