from flask import Flask, request, render_template_string, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

HTML = """(너가 이미 쓰던 HTML 그대로)"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()
    row = np.zeros(len(feature_names))

    for i, col in enumerate(feature_names):
        if col == "SEQN":
            row[i] = 92000
        elif col == "bmi_glu":
            row[i] = float(d["BMI"]) * float(d["Glucose"])
        elif col == "age_sbp":
            row[i] = float(d["Age"]) * float(d["SBP_mean"])
        elif col == "htn_dm":
            row[i] = float(d["Hypertension"]) + float(d["Diabetes"])
        elif col in d:
            row[i] = float(d[col])

    prob = model.predict_proba(scaler.transform([row]))[0][1]
    prob = prob ** 0.25

    return jsonify({"prob": float(prob)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
