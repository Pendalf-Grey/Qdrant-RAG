import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Импортируем основную функцию обработки запроса
from search_engine import process_query

# ====================== ЗАГРУЗКА ТОКЕНА ИЗ .ENV ======================
load_dotenv()  # ищет файл .env в текущей директории и загружает переменные

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Токен Telegram не найден! Создайте файл .env с TELEGRAM_TOKEN=ваш_токен")

# ====================== ЛОГИРОВАНИЕ ======================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ====================== ОБРАБОТЧИКИ ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для поиска по базе данных. Отправьте мне вопрос, и я постараюсь найти ответ."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    chat_id = update.effective_chat.id
    logger.info(f"Получено сообщение от {chat_id}: {user_query}")

    # Отправляем уведомление о начале обработки
    processing_msg = await update.message.reply_text("⏳ Обрабатываю запрос, пожалуйста, подождите...")

    try:
        # Вызываем асинхронную функцию из search_engine
        answer = await process_query(user_query)
        await processing_msg.delete()
        await update.message.reply_text(answer)
    except Exception as e:
        logger.exception("Ошибка при обработке запроса")
        await processing_msg.delete()
        await update.message.reply_text("❌ Произошла ошибка при обработке запроса. Попробуйте позже.")

# ====================== ЗАПУСК ======================

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен...")
    application.run_polling()

if __name__ == "__main__":
    main()