import pickle
from config.configurations import app
from keras.models import load_model

print(app.config["DEEP_LEARNING_MODEL_FOLDER"])
tokenizer = pickle.load(open(
    app.config["DEEP_LEARNING_MODEL_FOLDER"] + "/tokenizer.pickle", "rb"))

print(tokenizer)

model = load_model(
    app.config["DEEP_LEARNING_MODEL_FOLDER"] + "/model.h5")

print(model.summary())



