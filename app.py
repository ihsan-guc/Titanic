import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.htm")

@app.route("/predict", methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(str(int_features))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    predictionValue = str(prediction)
    print(str(prediction))
    if str(prediction) == "[1.]":
        output = "Hayatda Kalacaksınız";
    if str(prediction) == "[0.]":
        output = "Öleceksiniz";

    return render_template("index.htm", prediction_text = "{}".format(output) + "");
if __name__ == "__main__":
    app.run(debug = True)