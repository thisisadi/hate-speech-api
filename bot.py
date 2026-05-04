import os
import logging
import requests
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
API_URL = os.environ.get("HATE_SPEECH_API_URL", "http://localhost:5000")


def call_api(comment: str) -> dict:
    resp = requests.post(
        f"{API_URL}/predict",
        json={"comment": comment},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def format_response(result: dict, original_text: str) -> str:
    if result.get("error"):
        return f"Error: {result['error']}"

    is_toxic = result["is_toxic"]
    flagged = result["flagged_categories"]
    scores = result["scores"]

    if is_toxic:
        verdict = "TOXIC CONTENT DETECTED"
        categories = ", ".join(flagged)
        lines = [
            f"VERDICT: {verdict}",
            f"Categories: {categories}",
            "",
            "Scores:",
        ]
    else:
        verdict = "CLEAN"
        lines = [
            f"VERDICT: {verdict}",
            "",
            "Scores:",
        ]

    for label, info in scores.items():
        flag = "YES" if info["flagged"] else "no"
        prob = info["probability"]
        threshold = info["threshold"]
        bar = "" if info["flagged"] else ""
        lines.append(f"  {bar} {label:<15} {prob:>6.1f}%  (threshold: {threshold:.0f}%)")

    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "Hate Speech Detection Bot\n\n"
        "Send me any text and I will classify it across 6 toxicity categories:\n"
        "  toxic, severe_toxic, obscene, threat, insult, identity_hate\n\n"
        "Commands:\n"
        "  /start  Show this message\n"
        "  /help   Show usage examples\n"
        "  /about  About this project\n\n"
        "Just send any message to get a prediction."
    )
    await update.message.reply_text(welcome)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Usage Examples:\n\n"
        "Send any text message directly. The bot will return:\n"
        "  - Overall verdict (CLEAN or TOXIC)\n"
        "  - Which categories were flagged\n"
        "  - Probability scores for all 6 categories\n\n"
        "The bot uses a multi-label Logistic Regression model trained on\n"
        "2,009,376 comments across 4 datasets (Jigsaw, HateXplain,\n"
        "Twitter Hate Speech, and Civil Comments).\n\n"
        "Thresholds:\n"
        "  toxic: 50%  |  severe_toxic: 80%  |  obscene: 50%\n"
        "  threat: 70%  |  insult: 50%  |  identity_hate: 65%"
    )
    await update.message.reply_text(help_text)


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = (
        "About This Project\n\n"
        "This bot is the serving layer of a Big Data hate speech detection\n"
        "pipeline built as part of CS-GY 6513 Big Data at NYU Tandon.\n\n"
        "Architecture:\n"
        "  Training: Apache Spark 3.5.3 on a 5-node GCP Dataproc cluster\n"
        "  Dataset: 2M+ comments (Jigsaw + HateXplain + Twitter + Civil Comments)\n"
        "  Models: 6 TF-IDF + Logistic Regression classifiers (Spark MLlib)\n"
        "  API: Flask on Render.com (Python, PySpark local mode)\n"
        "  Bot: python-telegram-bot v20\n\n"
        "GitHub: https://github.com/pragya2002/hate-speech-detection\n\n"
        "Team: Aditya Jha, Pragya Awasthi, Tharun Murugesan"
    )
    await update.message.reply_text(about_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    comment = update.message.text.strip()
    if not comment:
        return

    # Send typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        result = call_api(comment)
        response = format_response(result, comment)
        await update.message.reply_text(response, parse_mode=None)
    except requests.exceptions.Timeout:
        await update.message.reply_text(
            "The API is warming up (cold start). Please try again in 30 seconds."
        )
    except Exception as e:
        logger.error(f"Error calling API: {e}")
        await update.message.reply_text(
            "Something went wrong. Please try again in a moment."
        )


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """In group chats, only respond when bot is mentioned or replied to."""
    message = update.message
    if not message or not message.text:
        return

    bot_username = context.bot.username
    is_mentioned = f"@{bot_username}" in message.text
    is_reply_to_bot = (
        message.reply_to_message and
        message.reply_to_message.from_user and
        message.reply_to_message.from_user.username == bot_username
    )

    if not (is_mentioned or is_reply_to_bot):
        return

    # Remove the bot mention from the text before classifying
    comment = message.text.replace(f"@{bot_username}", "").strip()
    if not comment:
        await message.reply_text("Please include a message to classify.")
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        result = call_api(comment)
        response = format_response(result, comment)
        await message.reply_text(response, parse_mode=None)
    except requests.exceptions.Timeout:
        await message.reply_text(
            "API is warming up. Please try again in 30 seconds."
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        await message.reply_text("Something went wrong. Please try again.")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))

    # Private chat: respond to all messages
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        handle_message
    ))

    # Group chat: respond only when mentioned or replied to
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
        handle_group_message
    ))

    logger.info("Bot started. Polling for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
