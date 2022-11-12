from config.configurations import app, db
from controllers.csv_routes import (
    view_columns,
    analyze_save_csv,
    getting_all_data_from_csv,
    getting_top_department_overall,
    getting_top_professor_overall,
    getting_top_professor_by_file, getting_top_department_by_file
)
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
    update_user_password,
    update_user_personal_info,
    update_user_security_info,
    update_user_username,
    verify_remove_account_token,
    verify_security_code,
)

# @desc: CSV routes for uploading csv files
app.add_url_rule("/data/view-columns",
                 view_func=view_columns, methods=["POST"])
app.add_url_rule("/data/analyze-save-csv",
                 view_func=analyze_save_csv, methods=["POST"])
app.add_url_rule("/data/get-all-data-from-csv",
                 view_func=getting_all_data_from_csv, methods=["GET"])
app.add_url_rule("/data/get-top-department-overall",
                 view_func=getting_top_department_overall, methods=["GET"])
app.add_url_rule("/data/get-top-professors-overall",
                 view_func=getting_top_professor_overall, methods=["GET"])
app.add_url_rule("/data/get-top-department-by-file/<page>",
                 view_func=getting_top_department_by_file, methods=["GET"])
app.add_url_rule("/data/get-top-professors-by-file/<page>",
                 view_func=getting_top_professor_by_file, methods=["GET"])
# @desc: User routes for authentication
app.add_url_rule("/user/authenticate",
                 view_func=authenticate, methods=["POST"])
app.add_url_rule("/user/check-email",
                 view_func=check_email, methods=["POST"])
app.add_url_rule("/user/checkpoint-2fa",
                 view_func=send_security_code, methods=["POST"])
app.add_url_rule("/user/forgot-password",
                 view_func=forgot_password, methods=["POST"])
app.add_url_rule("/user/get_user",
                 view_func=get_authenticated_user, methods=["GET"])
app.add_url_rule("/user/remove-email-from-account",
                 view_func=remove_email_from_account, methods=["POST"])
app.add_url_rule("/user/reset-password/<token>",
                 view_func=reset_password, methods=["POST"])
app.add_url_rule("/user/sign-out",
                 view_func=signout, methods=["POST"])
app.add_url_rule("/user/signup",
                 view_func=signup, methods=["POST"])
app.add_url_rule("/user/update-password",
                 view_func=update_user_password, methods=["PUT"])
app.add_url_rule("/user/update-personal-info",
                 view_func=update_user_personal_info, methods=["PUT"])
app.add_url_rule("/user/update-security-info",
                 view_func=update_user_security_info, methods=["PUT"])
app.add_url_rule("/user/update-username",
                 view_func=update_user_username, methods=["PUT"])
app.add_url_rule("/user/verify-2fa",
                 view_func=verify_security_code, methods=["POST"])
app.add_url_rule("/user/verify-remove-account-token/<token>",
                 view_func=verify_remove_account_token, methods=["GET"])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    db.create_all()
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
