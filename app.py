from config.configurations import app, db
from controllers.user_routes import (signup,
                                     authenticate,
                                     get_authenticated_user,
                                     signout,
                                     check_email,
                                     forgot_password,
                                     reset_password)

# @desc: User routes for authentication
app.add_url_rule("/signup", view_func=signup, methods=["POST"])
app.add_url_rule("/authenticate", view_func=authenticate, methods=["POST"])
app.add_url_rule("/get_user", view_func=get_authenticated_user, methods=["GET"])
app.add_url_rule("/sign-out", view_func=signout, methods=["GET"])
app.add_url_rule("/check-email", view_func=check_email, methods=["POST"])
app.add_url_rule("/forgot-password", view_func=forgot_password, methods=["POST"])
app.add_url_rule("/reset-password/<token>", view_func=reset_password, methods=["POST"])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    db.create_all()
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
