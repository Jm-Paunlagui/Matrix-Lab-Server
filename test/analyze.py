# import os
# import pickle
#
# import pandas as pd
# from keras.utils import pad_sequences
#
# from config.configurations import app
# from keras.models import load_model
#
# from database_queries.csv_queries import csv_formatter
#
# file_name = "eval_raw.csv"
#
# # user will enter the index of the columns that will be renamed to the required format in the csv file
# sentence_index = 4
# evaluatee_index = 0
# department_index = 1
# course_code_index = 2
# csv_question = "What are the strengths of the instructor in teaching the course?"
# school_year = "S.Y. 2022-2023"
#
# # @desc: Format the csv file to the required format: sentence, evaluatee, department and course code.
# csv_formatter(file_name, sentence_index, evaluatee_index, department_index, course_code_index)
#
# # @desc: Read the reformatted csv file and return a pandas dataframe object
# csv_to_pred = pd.read_csv(app.config["CSV_REFORMATTED_FOLDER"] + "/" + file_name)
#
# # remove the rows that have empty values in the sentence column or any other column that is required for the evaluation
# csv_to_pred = csv_to_pred.dropna(subset=["sentence"])
#
# tokenizer = pickle.load(open(
#     app.config["DEEP_LEARNING_MODEL_FOLDER"] + "/tokenizer.pickle", "rb"))
#
# model = load_model(
#     app.config["DEEP_LEARNING_MODEL_FOLDER"] + "/model.h5")
#
# # @desc: Get the sentences from the csv file
# sentences = csv_to_pred["sentence"].to_list()
#
# # @desc: Lowercase
# sentences = [sentence.lower() for sentence in sentences]
#
# # @desc: Tokenize the sentences
# tokenized_sentences = tokenizer.texts_to_sequences(sentences)
#
# # @desc: Pad the tokenized sentences
# padded_sentences = pad_sequences(tokenized_sentences, maxlen=300, padding='post')
#
# # @desc: Predict the sentiment of the sentences
# predictions = model.predict(padded_sentences)
#
# predictions = [round(round(prediction[0], 4) * 100, 2) for prediction in predictions]
#
# # @desc: Add the predictions to the csv file
# csv_to_pred["sentiment"] = predictions
#
# school_year = school_year.replace("S.Y.", "SY").replace(" ", "")
# csv_question = csv_question.title()
# csv_question = csv_question.replace("?", "")
# csv_question = csv_question.replace(" ", "_")
# csv_to_pred.to_csv(
#     app.config["CSV_ANALYZED_FOLDER"] + "/" + "ANALYZED-" + csv_question + "_" + school_year + ".csv", index=False)
# # @desc: Delete the reformatted csv file from the reformatted folder
# os.remove(os.path.join(app.config["CSV_REFORMATTED_FOLDER"], file_name))


