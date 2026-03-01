"""Telegram channel adapter for LiteAgent — polling + webhook modes."""

import logging

logger = logging.getLogger(__name__)

# Max message length in Telegram
TG_MAX_LENGTH = 4096


def _make_message_handler(agent):
    """Create reusable message handler for both polling and webhook modes."""
    async def handle_message(update, context):
        if not update.message or not update.message.text:
            return
        user_id = f"tg-{update.effective_user.id}"
        text = update.message.text

        # Check slash commands first
        cmd_response = agent.handle_command(text, user_id)
        if cmd_response is not None:
            await update.message.reply_text(cmd_response)
            return

        # Run agent
        try:
            response = await agent.run(text, user_id)
            for i in range(0, len(response), TG_MAX_LENGTH):
                await update.message.reply_text(response[i:i + TG_MAX_LENGTH])
        except Exception as e:
            logger.error("Error handling Telegram message: %s", e)
            await update.message.reply_text(f"Error: {e}")

    return handle_message


async def run_telegram(agent, config: dict):
    """Run Telegram bot — dispatches to polling or webhook based on config."""
    mode = config.get("mode", "polling")
    if mode == "webhook":
        raise ValueError(
            "Webhook mode requires FastAPI. Use --channel api with "
            "telegram webhook_url configured, or use polling mode."
        )
    await _run_polling(agent, config)


async def _run_polling(agent, config: dict):
    """Run Telegram bot with long-polling."""
    try:
        from telegram import Update
        from telegram.ext import Application, MessageHandler, filters
    except ImportError:
        raise ImportError(
            "python-telegram-bot is required: pip install liteagent[telegram]"
        )

    token = config.get("token")
    if not token:
        raise ValueError("Telegram token not configured. Set token_env in channels.telegram config.")

    handler = _make_message_handler(agent)
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler))

    logger.info("Telegram bot started (polling mode)")
    await app.run_polling()


def setup_webhook_route(app, agent, tg_config: dict):
    """Mount Telegram webhook route on a FastAPI app."""
    try:
        from telegram import Update
        from telegram.ext import Application, MessageHandler, filters
    except ImportError:
        raise ImportError(
            "python-telegram-bot is required: pip install liteagent[telegram]"
        )

    token = tg_config.get("token")
    webhook_url = tg_config.get("webhook_url")
    webhook_secret = tg_config.get("webhook_secret", "")

    if not token or not webhook_url:
        raise ValueError("Telegram webhook requires token and webhook_url in config")

    # Build telegram Application for update processing
    tg_app = Application.builder().token(token).build()
    handler = _make_message_handler(agent)
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler))

    @app.on_event("startup")
    async def _setup_webhook():
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.bot.set_webhook(
            url=f"{webhook_url}/telegram/webhook",
            secret_token=webhook_secret)
        logger.info("Telegram webhook set to %s/telegram/webhook", webhook_url)

    @app.on_event("shutdown")
    async def _cleanup_webhook():
        await tg_app.stop()
        await tg_app.shutdown()

    @app.post("/telegram/webhook")
    async def telegram_webhook(request):
        """Receive Telegram webhook updates."""
        body = await request.json()
        update = Update.de_json(body, tg_app.bot)
        await tg_app.process_update(update)
        return {"ok": True}

    logger.info("Telegram webhook route mounted")
