# Hate Speech Detection API

A real-time REST API and Telegram bot for multi-label hate speech detection, trained on 2,009,376 comments across 4 datasets using Apache Spark MLlib on a 5-node GCP Dataproc cluster.

## Live Demo

- **API**: https://hate-speech-api.onrender.com
- **Telegram Bot**: @HateSpeechDetectorBot

## Architecture

```
Telegram Message
      ↓
Telegram Bot (python-telegram-bot)
      ↓
Flask REST API (Render.com, HTTPS)
      ↓
6 PySpark PipelineModel classifiers (local mode)
      ↓
Multi-label prediction response
```

## API Usage

### Health check
```bash
GET https://hate-speech-api.onrender.com/
```

### Predict (POST)
```bash
curl -X POST https://hate-speech-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "your text here"}'
```

### Predict (GET)
```bash
curl "https://hate-speech-api.onrender.com/predict?comment=your+text+here"
```

### Response format
```json
{
  "comment": "original comment",
  "cleaned": "preprocessed text",
  "is_toxic": true,
  "flagged_categories": ["toxic", "insult"],
  "scores": {
    "toxic":         {"probability": 63.76, "threshold": 50.0, "flagged": true},
    "severe_toxic":  {"probability": 52.28, "threshold": 80.0, "flagged": false},
    "obscene":       {"probability": 43.53, "threshold": 50.0, "flagged": false},
    "threat":        {"probability": 45.71, "threshold": 70.0, "flagged": false},
    "insult":        {"probability": 65.06, "threshold": 50.0, "flagged": true},
    "identity_hate": {"probability": 41.86, "threshold": 65.0, "flagged": false}
  }
}
```

## Models

6 independent Spark MLlib Pipeline models, each with:
- RegexTokenizer → StopWordsRemover → HashingTF (10K features) → IDF → LogisticRegression

| Label | AUC-ROC | Threshold |
|---|---|---|
| toxic | 0.857 | 50% |
| severe_toxic | 0.928 | 80% |
| obscene | 0.916 | 50% |
| threat | 0.919 | 70% |
| insult | 0.884 | 50% |
| identity_hate | 0.929 | 65% |

## Local Development

```bash
# Clone the repo
git clone https://github.com/thisisadi/hate-speech-api
cd hate-speech-api

# Install dependencies (requires Java 11+)
pip install -r requirements.txt

# Run the Flask API
python app.py

# In a separate terminal, run the Telegram bot
export TELEGRAM_BOT_TOKEN=your_token_here
export HATE_SPEECH_API_URL=http://localhost:5000
python bot.py
```

## Deployment (Render)

1. Fork this repo
2. Connect to Render.com
3. Create a new Web Service from the repo (Docker)
4. Add environment variable: `PORT=5000`
5. Create a Background Worker for the bot
6. Add environment variables: `TELEGRAM_BOT_TOKEN` and `HATE_SPEECH_API_URL`

## Training Infrastructure

The models were trained on NYU Dataproc (GCP):
- 5-node YARN cluster (Hadoop 3.3.6, Spark 3.5.3)
- 2,009,376 training examples across Jigsaw, HateXplain, Twitter Hate Speech, and Civil Comments
- Sub-linear scaling: 2.1x time for 4x data
- Full pipeline: https://github.com/pragya2002/hate-speech-detection

## Related

- Main project repo: https://github.com/pragya2002/hate-speech-detection
- Course: CS-GY 6513 Big Data, NYU Tandon, Spring 2026
