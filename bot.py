import os
import logging
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from telegram import Update, ChatPermissions
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

MAX_WARNINGS = 3
MUTE_DURATION_MINUTES = 30

# In-memory warning tracker: {chat_id: {user_id: warning_count}}
warnings = defaultdict(lambda: defaultdict(int))
# Track muted users: {chat_id: {user_id: mute_until_datetime}}
muted_until = defaultdict(dict)

CATEGORY_LABELS = {
    "toxic":         "toxic content",
    "severe_toxic":  "severely toxic content",
    "obscene":       "obscene language",
    "threat":        "threatening language",
    "insult":        "insults",
    "identity_hate": "identity-based hate speech",
}


def call_api(comment: str) -> dict:
    resp = requests.post(
        f"{API_URL}/predict",
        json={"comment": comment},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def get_flagged_description(flagged_categories: list) -> str:
    if not flagged_categories:
        return "harmful content"
    labels = [CATEGORY_LABELS.get(c, c) for c in flagged_categories]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hate Speech Detection Bot is active.\n\n"
        "I silently monitor messages and warn users who send harmful content. "
        "After 3 warnings, the user will be muted for 30 minutes.\n\n"
        "Commands:\n"
        "  /warnings  Check your current warning count\n"
        "  /about     About this bot"
    )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hate Speech Detection Bot\n\n"
        "Trained on 2,009,376 comments across 4 datasets using Apache Spark MLlib "
        "on a 5-node GCP cluster. Detects 6 categories of harmful content: "
        "toxic, severe toxic, obscene, threat, insult, and identity hate.\n\n"
        "Built by Aditya Jha, Pragya Awasthi, Tharun Murugesan\n"
        "CS-GY 6513 Big Data, NYU Tandon, Spring 2026\n\n"
        "API: https://hate-speech-api-lqrg.onrender.com"
    )


async def warnings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    count = warnings[chat_id][user_id]
    remaining = MAX_WARNINGS - count

    if count == 0:
        await update.message.reply_text("You have no warnings.")
    elif count >= MAX_WARNINGS:
        mute_time = muted_until[chat_id].get(user_id)
        if mute_time and datetime.now() < mute_time:
            mins_left = int((mute_time - datetime.now()).total_seconds() / 60) + 1
            await update.message.reply_text(
                f"You are currently muted for {mins_left} more minute(s)."
            )
        else:
            await update.message.reply_text(
                f"You have {count} warning(s). One more violation will result in a mute."
            )
    else:
        await update.message.reply_text(
            f"You have {count}/{MAX_WARNINGS} warning(s). "
            f"{remaining} more violation(s) will result in a 30-minute mute."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.text:
        return

    user = update.effective_user
    chat_id = update.effective_chat.id
    user_id = user.id
    username = f"@{user.username}" if user.username else user.first_name

    # Check if currently muted
    mute_time = muted_until[chat_id].get(user_id)
    if mute_time and datetime.now() < mute_time:
        # Already muted by Telegram — just ignore
        return

    # Clear expired mute
    if mute_time and datetime.now() >= mute_time:
        del muted_until[chat_id][user_id]
        warnings[chat_id][user_id] = 0

    comment = message.text.strip()

    try:
        result = call_api(comment)
    except requests.exceptions.Timeout:
        logger.warning("API timeout")
        return
    except Exception as e:
        logger.error(f"API error: {e}")
        return

    if not result.get("is_toxic"):
        # Clean message — stay silent
        return

    # Toxic message detected
    flagged = result.get("flagged_categories", [])
    description = get_flagged_description(flagged)
    warnings[chat_id][user_id] += 1
    count = warnings[chat_id][user_id]
    remaining = MAX_WARNINGS - count

    try:
        await message.delete()
    except Exception:
        pass  # Bot may not have delete permissions — continue anyway

    if count >= MAX_WARNINGS:
        # Mute the user
        mute_until_time = datetime.now() + timedelta(minutes=MUTE_DURATION_MINUTES)
        muted_until[chat_id][user_id] = mute_until_time

        try:
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=mute_until_time
            )
            await message.reply_text(
                f"{username} has been muted for {MUTE_DURATION_MINUTES} minutes "
                f"after receiving {MAX_WARNINGS} warnings for posting harmful content."
            )
            logger.info(f"Muted {username} (id={user_id}) in chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to mute user: {e}")
            await message.reply_text(
                f"{username} has reached {MAX_WARNINGS} warnings. "
                f"Please grant this bot admin rights to enforce mutes."
            )
    elif count == MAX_WARNINGS - 1:
        # Final warning
        await message.reply_text(
            f"Final warning {username}. Your message was removed for containing {description}. "
            f"One more violation will result in a {MUTE_DURATION_MINUTES}-minute mute."
        )
    else:
        # Regular warning
        await message.reply_text(
            f"Warning {count}/{MAX_WARNINGS} for {username}. "
            f"Your message was removed for containing {description}. "
            f"{remaining} more violation(s) will result in a {MUTE_DURATION_MINUTES}-minute mute."
        )

    logger.info(f"Warning {count}/{MAX_WARNINGS} issued to {username} for: {flagged}")


def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("warnings", warnings_command))

    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    logger.info("Bot started. Polling for messages...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
