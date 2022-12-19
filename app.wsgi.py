from app import app as application
import sys
sys.path.insert(
    0, 'C:\\Users\\paunl\\Jm-Paunlagui\\Pycharm-Projects\Matrix-Lab-Server')

activate_this = 'C:\\Users\\paunl\\Jm-Paunlagui\\Pycharm-Projects\Matrix-Lab-Server\\venv\Scripts\\activatte.bat'

with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))


if __name__ == "__main__":
    application.run()
