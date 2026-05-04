import os
import re
import logging
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, lower, regexp_replace, trim

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
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Spark session (local mode, no cluster needed) ─────────────────────────────
logger.info("Starting Spark session...")
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("HateSpeechAPI") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
logger.info("Spark session started.")

# ── Load all 6 models once at startup ────────────────────────────────────────
logger.info("Loading models...")
models = {}
for label in LABELS:
    model_path = os.path.join(MODELS_DIR, f"{label}_model")
    models[label] = PipelineModel.load(model_path)
    logger.info(f"  Loaded: {label}_model")
logger.info("All models loaded. API ready.")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(comment):
    cleaned = clean_text(comment)
    df = spark.createDataFrame([(cleaned,)], ["comment_text"])

    results = {}
    flagged = []

    for label in LABELS:
        pred = models[label].transform(df)
        row = pred.select("probability").collect()[0]
        prob = float(row["probability"][1])
        threshold = THRESHOLDS[label]
        is_toxic = prob >= threshold
        results[label] = {
            "probability": round(prob * 100, 2),
            "threshold": threshold * 100,
            "flagged": is_toxic
        }
        if is_toxic:
            flagged.append(label)

    return {
        "comment": comment,
        "cleaned": cleaned,
        "is_toxic": len(flagged) > 0,
        "flagged_categories": flagged,
        "scores": results
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Hate Speech Detection API",
        "models_loaded": list(models.keys()),
        "version": "1.0.0"
    })


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    if not data or "comment" not in data:
        return jsonify({"error": "Missing 'comment' field in request body"}), 400

    comment = data["comment"].strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    if len(comment) > 5000:
        return jsonify({"error": "Comment exceeds 5000 character limit"}), 400

    try:
        result = predict(comment)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


@app.route("/predict", methods=["GET"])
def predict_get():
    comment = request.args.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "Missing 'comment' query parameter"}), 400
    try:
        result = predict(comment)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
