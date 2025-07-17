import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)

    output = "Fraud" if prediction[0] == 1 else "Not Fraud"
    return render_template("index.html", prediction_text=f"Transaction is {output}")

if __name__ == "__main__":
    app.run(debug=True)
