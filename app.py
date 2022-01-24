# Import the required libraries
from flask import Flask, render_template, redirect, request
import string
import pickle
import os
port = int(os.environ.get("PORT", 5000))

#Loading vectoriser
vectorizor = pickle.load( open("Model_files/Vectorisors/log_reg_vec.pkl","rb"))
model = pickle.load(open("Model_files/Models/log_reg_model.pkl", 'rb'))

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
        data = [message]

        # feeds entered line into the model
        vect = vectorizor.transform(data).toarray()

        # stroes predicted value
        prediction = model.predict(vect)[0]
    return render_template("fk_index.html",outcome = prediction)


# main
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port =port, debug = True)
