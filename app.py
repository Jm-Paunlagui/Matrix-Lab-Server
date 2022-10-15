from config.configurations import app, db
import controllers.user_routes

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    db.create_all()
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
