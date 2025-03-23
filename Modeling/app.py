import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    prediction = model.predict(features)
    prediction_text = "Canceled 🤬" if prediction == 0 else "Not Canceled 😇"

    return prediction_text

if __name__ == "__main__":
    app.run(debug=True)