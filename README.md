# Hate Speech Detection API

A real-time REST API and Telegram bot for multi-label hate speech detection.

**Models trained on 2,009,376 comments** across 4 datasets using Apache Spark MLlib on a 5-node GCP Dataproc cluster. Weights extracted and served via pure NumPy — no JVM required, runs on 128MB RAM.

## Live Demo

- **API**: https://hate-speech-api-lqrg.onrender.com
- **Telegram Bot**: @HateSpeechDetectorBot

## API Usage

### Health check
```bash
curl https://hate-speech-api-lqrg.onrender.com/
```

### Predict (POST)
```bash
curl -X POST https://hate-speech-api-lqrg.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "your text here"}'
```

### Predict (GET)
```bash
curl "https://hate-speech-api-lqrg.onrender.com/predict?comment=your+text+here"
```

### Response
```json
{
  "comment": "original text",
  "cleaned": "preprocessed tokens",
  "is_toxic": false,
  "flagged_categories": [],
  "scores": {
    "toxic":         {"probability": 43.34, "threshold": 50.0, "flagged": false},
    "severe_toxic":  {"probability": 56.45, "threshold": 80.0, "flagged": false},
    "obscene":       {"probability": 43.53, "threshold": 50.0, "flagged": false},
    "threat":        {"probability": 45.83, "threshold": 70.0, "flagged": false},
    "insult":        {"probability": 40.03, "threshold": 50.0, "flagged": false},
    "identity_hate": {"probability": 39.85, "threshold": 65.0, "flagged": false}
  }
}
```

## Architecture

```
Telegram Message
      ↓
Telegram Bot (python-telegram-bot)
      ↓
Flask REST API (Render.com, HTTPS, 512MB RAM)
      ↓
Pure NumPy inference (no JVM, no Spark)
  - MurmurHash3 tokenization (mmh3)
  - TF-IDF weights loaded from model_weights.json
  - Logistic regression: sigmoid(coef · tfidf + intercept)
      ↓
Multi-label prediction response (<10ms latency)
```

## Models

6 independent TF-IDF + Logistic Regression classifiers trained with Spark MLlib.
Weights extracted to `model_weights.json` (1.4MB) for lightweight serving.

| Label | AUC-ROC | Threshold |
|---|---|---|
| toxic | 0.857 | 50% |
| severe_toxic | 0.928 | 80% |
| obscene | 0.916 | 50% |
| threat | 0.919 | 70% |
| insult | 0.884 | 50% |
| identity_hate | 0.929 | 65% |

## Training Infrastructure

- 5-node YARN cluster (Hadoop 3.3.6, Spark 3.5.3) on NYU Dataproc (GCP)
- 2,009,376 training examples: Jigsaw + HateXplain + Twitter + Civil Comments
- Full pipeline repo: https://github.com/pragya2002/hate-speech-detection

## Local Development

```bash
git clone https://github.com/thisisadi/hate-speech-api
cd hate-speech-api
pip install -r requirements.txt

# Run API
python app.py

# Run bot (separate terminal)
export TELEGRAM_BOT_TOKEN=your_token
export HATE_SPEECH_API_URL=http://localhost:5000
python bot.py
```
