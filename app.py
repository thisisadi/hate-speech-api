import os
import re
import json
import logging
import numpy as np
import mmh3
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
THRESHOLDS = {
    "toxic":         0.50,
    "severe_toxic":  0.80,
    "obscene":       0.50,
    "threat":        0.70,
    "insult":        0.50,
    "identity_hate": 0.65,
}
NUM_FEATURES = 10000
BASE_DIR = os.path.dirname(__file__)

# ── Load weights and stop words once at startup ───────────────────────────────
logger.info("Loading model weights...")
with open(os.path.join(BASE_DIR, "model_weights.json")) as f:
    raw = json.load(f)

WEIGHTS = {
    label: {
        "coef":      np.array(w["coef"],      dtype=np.float64),
        "idf":       np.array(w["idf"],       dtype=np.float64),
        "intercept": float(w["intercept"]),
    }
    for label, w in raw.items()
}

with open(os.path.join(BASE_DIR, "stopwords.json")) as f:
    STOP_WORDS = set(json.load(f))

logger.info(f"Loaded {len(WEIGHTS)} models. API ready.")


# ── Inference (pure Python + numpy, no JVM) ───────────────────────────────────

def hash_token(token: str) -> int:
    """Spark-compatible MurmurHash3."""
    return abs(mmh3.hash(token, 0, signed=True)) % NUM_FEATURES


def hashing_tf(tokens: list) -> np.ndarray:
    """Replicate Spark HashingTF(numFeatures=10000)."""
    freq = np.zeros(NUM_FEATURES, dtype=np.float64)
    for t in tokens:
        freq[hash_token(t)] += 1.0
    return freq


def preprocess(text: str) -> list:
    """Replicate Spark pipeline: lower → strip URLs → strip non-alpha → tokenize → remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.split(r"\W+", text)
    return [t for t in tokens if t and t not in STOP_WORDS]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict(comment: str) -> dict:
    tokens = preprocess(comment)
    tf = hashing_tf(tokens)

    scores = {}
    flagged = []

    for label in LABELS:
        w = WEIGHTS[label]
        tfidf = tf * w["idf"]
        raw_score = float(np.dot(w["coef"], tfidf)) + w["intercept"]
        prob = sigmoid(raw_score)
        threshold = THRESHOLDS[label]
        is_flagged = prob >= threshold

        scores[label] = {
            "probability": round(prob * 100, 2),
            "threshold":   threshold * 100,
            "flagged":     is_flagged,
        }
        if is_flagged:
            flagged.append(label)

    return {
        "comment":           comment,
        "cleaned":           " ".join(tokens),
        "is_toxic":          len(flagged) > 0,
        "flagged_categories": flagged,
        "scores":            scores,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "service":       "Hate Speech Detection API",
        "models_loaded": LABELS,
        "version":       "2.0.0",
        "backend":       "numpy (no PySpark)",
    })


@app.route("/predict", methods=["POST"])
def predict_post():
    data = request.get_json()
    if not data or "comment" not in data:
        return jsonify({"error": "Missing 'comment' field in request body"}), 400
    comment = data["comment"].strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    if len(comment) > 5000:
        return jsonify({"error": "Comment exceeds 5000 character limit"}), 400
    return jsonify(predict(comment))


@app.route("/predict", methods=["GET"])
def predict_get():
    comment = request.args.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "Missing 'comment' query parameter"}), 400
    return jsonify(predict(comment))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
