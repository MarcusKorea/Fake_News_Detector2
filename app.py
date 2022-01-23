# Import the required libraries
from flask import Flask, render_template, redirect, request
import spacy
import nltk
from nltk.tokenize import word_tokenize
import string
import pickle
import os
port = int(os.environ.get("PORT", 5000))

#Loading vectoriser
vectorizor = pickle.load( open("Model_files/Vectorisors/log_reg_vec.pkl","rb"))
model = pickle.load(open("Model_files/Models/log_reg_model.pkl", 'rb'))


# define some user functions
# remove non ascii characters
def remove_non_ascii(word):
    new_word = word.encode("ascii","ignore").decode()
    return new_word
    # remove punctuation and stop words
def clean_input(input):
    # load language model
    sp = spacy.load('en_core_web_sm')
    # import stop words
    all_stopwords = sp.Defaults.stop_words
    # import puncuation
    punc  = string.punctuation

    # replace any weird characters
    text = input
    remove_non_ascii(text)

    # remove puncuation
    new_text = text.translate(str.maketrans('', '', punc))

    # tokenize and make everything lower case
    text_tokens = word_tokenize(new_text.lower())

    # remove stop words
    tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]

    # make list a string again
    tokens_without_sw = " ".join(tokens_without_sw)
    return tokens_without_sw

# Start Flask
app =Flask(__name__)

# home page
@app.route("/",  methods=["GET","POST"])
def index():
    print("in index)")
    print( request.method)
    if request.method == "GET":
        return render_template("fk_index.html")


    if request.method == "POST":
        article = request.form["Contents"]
        return render_template("fk_index.html")

@app.route('/predict',methods=['POST'])
def predict():
    print("in predict")
    if request.method == 'POST':
        message = [str(x) for x in request.form.values()][0]
        # removes stop word, punctuation and non ascci characters from entered line
        message = clean_input(message)
        data = [message]

        # feeds entered line into the model
        vect = vectorizor.transform(data).toarray()

        # stroes predicted value
        prediction = model.predict(vect)[0]
    return render_template("fk_index.html",outcome = prediction)


# main
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port =port, debug = True)
