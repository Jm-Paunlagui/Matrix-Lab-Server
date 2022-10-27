from config.configurations import app, db
from controllers.user_routes import (
    authenticate,
    check_email,
    forgot_password,
    get_authenticated_user,
    remove_email_from_account,
    reset_password,
    send_security_code,
    signout,
    signup,
    verify_remove_account_token, 
    verify_security_code,
    )

# @desc: User routes for authentication
app.add_url_rule("/user/authenticate", view_func=authenticate, methods=["POST"])
app.add_url_rule("/user/check-email", view_func=check_email, methods=["POST"])
app.add_url_rule("/user/checkpoint-2fa", view_func=send_security_code, methods=["POST"])
app.add_url_rule("/user/forgot-password", view_func=forgot_password, methods=["POST"])
app.add_url_rule("/user/get_user", view_func=get_authenticated_user, methods=["GET"])
app.add_url_rule("/user/remove-email-from-account", view_func=remove_email_from_account, methods=["POST"])
app.add_url_rule("/user/reset-password/<token>", view_func=reset_password, methods=["POST"])
app.add_url_rule("/user/sign-out", view_func=signout, methods=["POST"])
app.add_url_rule("/user/signup", view_func=signup, methods=["POST"])
app.add_url_rule("/user/verify-2fa", view_func=verify_security_code, methods=["POST"])
app.add_url_rule("/user/verify-remove-account-token/<token>", view_func=verify_remove_account_token, methods=["GET"])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    db.create_all()
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
