import os, joblib
from flask import Flask, request, render_template, jsonify
import os
from src.utils import get_paths 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

MODEL_PATH, VEC_PATH, METRICS_PATH = get_paths()
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    text = request.form.get("text") or (request.json.get("text") if request.is_json else None)
    if not text:
        return jsonify({"error": "Debe enviar 'text'"}), 400
    model = load_model()
    pred = model.predict([text])[0]
    proba = float(model.predict_proba([text])[0][1])
    if request.is_json:
        return jsonify({"label": pred, "spam_proba": proba})
    return render_template("result.html", text=text, label=pred, spam_proba=proba)

if __name__ == "__main__":
    app.run(debug=True)

