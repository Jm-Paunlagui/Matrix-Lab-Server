from config.configurations import app, db
from controllers.csv_routes import (
    view_columns,
    analyze_save_csv,
    getting_all_data_from_csv,
    getting_top_department_overall,
    getting_top_professor_overall,
    getting_top_professor_by_file, getting_top_department_by_file, options_for_file_data, getting_list_of_csv_files,
    viewing_csv_file, deleting_csv_file, downloading_csv_file, list_of_csv_files_to_view, reading_csv_file
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
    verify_security_code, verify_reset_password_token,
)
from testpy.analyze import get_top_professors_by_file

# @desc: CSV routes for uploading csv files
app.add_url_rule("/data/view-columns",
                 view_func=view_columns, methods=["POST"])
app.add_url_rule("/data/analyze-save-csv",
                 view_func=analyze_save_csv, methods=["POST"])
# @desc: Dashboard data
app.add_url_rule("/data/get-all-data-from-csv",
                 view_func=getting_all_data_from_csv, methods=["GET"])
app.add_url_rule("/data/get-top-department-overall",
                 view_func=getting_top_department_overall, methods=["GET"])
app.add_url_rule("/data/get-top-professors-overall",
                 view_func=getting_top_professor_overall, methods=["GET"])
app.add_url_rule("/data/options-for-file",
                 view_func=options_for_file_data, methods=["GET"])
app.add_url_rule("/data/get-top-department-by-file",
                 view_func=getting_top_department_by_file, methods=["POST"])
app.add_url_rule("/data/get-top-professors-by-file",
                 view_func=getting_top_professor_by_file, methods=["POST"])
app.add_url_rule("/data/list-of-csv-files-to-view/<int:page>",
                 view_func=getting_list_of_csv_files, methods=["GET"])
app.add_url_rule("/data/view-csv-file/<int:csv_id>",
                 view_func=viewing_csv_file, methods=["GET"])
# @desc: Delete csv file
app.add_url_rule("/data/delete-csv-file/<int:csv_id>",
                 view_func=deleting_csv_file, methods=["DELETE"])
# @desc: Download csv file
app.add_url_rule("/data/download-csv-file/<int:csv_id>",
                 view_func=downloading_csv_file, methods=["GET"])
# desc: To there directory to view the csv file
app.add_url_rule("/data/get-list-of-taught-courses/<int:csv_id>/<string:folder_name>",
                 view_func=list_of_csv_files_to_view, methods=["GET"])
app.add_url_rule("/data/read-data-response/<int:csv_id>/<string:folder_name>/<string:file_name>",
                 view_func=reading_csv_file, methods=["GET"])
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
app.add_url_rule("/user/verify-reset-password-token/<token>",
                 view_func=verify_reset_password_token, methods=["GET"])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    db.create_all()
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
