#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HealthMateBot: An AI General Health & Wellness Telegram Bot
This bot uses ONE single AI model to manage all interactions,
and it stores user data in a cloud-hosted Supabase (PostgreSQL) database.
"""

import logging
import os
import json
import re
import datetime

from dotenv import load_dotenv
import google.generativeai as genai

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ChatAction

from supabase import create_client, Client  # Supabase client

# ------------------ LOAD ENV ------------------
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# ------------------ AI SYSTEM INSTRUCTION (HEALTH) ------------------
MASTER_SYSTEM_INSTRUCTION = (
    "You are HealthMate, a friendly AI for general health and wellness. "
    "You are NOT a doctor and you MUST NOT diagnose serious conditions. "
    "You give simple advice about: daily health, diet, sleep, hydration, "
    "mild fever/cold care, lifestyle tips, and general wellness.\n"
    "\n"
    "--- BEHAVIOR RULES ---\n"
    "1. NEW USER: If we tell you the user is NEW, introduce yourself and ask these questions "
    "ONE by ONE: 1) Name, 2) Age, 3) Gender, 4) Height (cm), 5) Weight (kg), "
    "6) Activity level (Sedentary/Light/Moderate/Active), 7) Any recurring health issue.\n"
    "2. RETURNING USER: If we give you their profile and last plan, greet them by name, "
    "briefly remind them of their last wellness plan, and ask how you can help today.\n"
    "3. GENERAL QUESTIONS: If the user just asks something like 'what to eat in fever', "
    "'how much water per day', 'how to sleep better', simply answer. Do NOT force the full setup.\n"
    "4. NEW / UPDATED PLAN: If they ask for a full 'plan', 'daily routine', 'diet plan', "
    "or confirm they want an updated plan after new info, you must generate a 10-point plan.\n"
    "5. PROFILE UPDATES: If user says 'I lost 3kg', 'now I am 21', etc., treat it as updated "
    "profile info, acknowledge it, and ask if they want a new plan.\n"
    "6. EMERGENCY / SERIOUS: If they describe chest pain, difficulty breathing, heavy bleeding, "
    "high fever for many days, confusion, or anything severe, you MUST say you cannot help and "
    "insist they contact a real doctor or emergency service.\n"
    "\n"
    "--- OUTPUT RULES ---\n"
    "RULE A – When giving a wellness PLAN:\n"
    "• First line MUST be: [USER_DATA_JSON] { ...latest profile... }\n"
    "• Then EXACTLY 10 numbered points (1 to 10) with wellness/diet/lifestyle suggestions.\n"
    "• Then a medical disclaimer like: 'This is general advice, not medical treatment...'\n"
    "• The very last token of the message MUST be [END_OF_PLAN].\n"
    "\n"
    "RULE B – For simple Q&A (no plan):\n"
    "• DO NOT use [USER_DATA_JSON] or [END_OF_PLAN].\n"
)

# ------------------ LOGGING ------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class HealthCareBot:
    """All bot logic is inside this class."""

    def __init__(self, telegram_token: str, gemini_api_key: str):
        self.telegram_token = telegram_token
        self.gemini_api_key = gemini_api_key

        # --- Supabase client ---
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            logger.error("Supabase URL / SERVICE_KEY missing in .env")
            self.supabase = None
        else:
            try:
                self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
                logger.info("Connected to Supabase.")
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}", exc_info=True)
                self.supabase = None

        # --- Gemini model ---
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=MASTER_SYSTEM_INSTRUCTION,
            )
            logger.info("Gemini model configured.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}", exc_info=True)
            self.model = None

    # --------- Helpers for Supabase ----------

    def _load_profile(self, user_id: int):
        """
        Load profile + last active plan from Supabase.
        If anything fails, return None and treat as new user.
        """
        if not self.supabase:
            return None

        try:
            profile_res = (
                self.supabase.table("users")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if not profile_res.data:
                return None

            profile = profile_res.data[0]

            plan_res = (
                self.supabase.table("plan_history")
                .select("plan_text")
                .eq("user_id", user_id)
                .is_("end_date", None)
                .order("start_date", desc=True)
                .limit(1)
                .execute()
            )

            last_plan = "No previous plan found."
            if plan_res.data:
                last_plan = plan_res.data[0]["plan_text"]

            return {"profile": profile, "last_plan": last_plan}
        except Exception as e:
            logger.error(f"Error loading profile for {user_id}: {e}", exc_info=True)
            return None

    def _log_conversation(self, user_id: int, sender: str, message: str):
        if not self.supabase:
            return
        try:
            self.supabase.table("conversation_history").insert(
                {
                    "user_id": user_id,
                    "sender": sender,
                    "message_text": message,
                }
            ).execute()
        except Exception as e:
            logger.error(f"Error logging conversation: {e}", exc_info=True)

    def _extract_profile_json(self, text: str):
        match = re.search(r"\[USER_DATA_JSON\]\s*(\{.*?\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.error("Failed to parse USER_DATA_JSON")
        return None

    def _extract_plan_text(self, text: str) -> str:
        try:
            plan_text = text.replace("[END_OF_PLAN]", "").strip()

            json_match = re.search(
                r"\[USER_DATA_JSON\]\s*(\{.*?\})", plan_text, re.DOTALL
            )
            if json_match:
                plan_text = plan_text.replace(json_match.group(0), "").strip()

            return plan_text.strip()
        except Exception as e:
            logger.error(f"Error extracting plan text: {e}", exc_info=True)
            return "Error: Could not extract plan."

    def save_new_plan_and_profile(self, user_id: int, profile: dict, plan: str):
        if not self.supabase or not profile:
            return

        now_iso = datetime.datetime.now().isoformat()
        profile_to_save = profile.copy()
        profile_to_save["user_id"] = user_id

        try:
            # upsert profile
            self.supabase.table("users").upsert(profile_to_save).execute()

            # close old active plan
            self.supabase.table("plan_history").update(
                {"end_date": now_iso}
            ).eq("user_id", user_id).is_("end_date", None).execute()

            # insert new plan
            self.supabase.table("plan_history").insert(
                {
                    "user_id": user_id,
                    "plan_text": plan,
                    "profile_json": profile,
                    "start_date": now_iso,
                    "end_date": None,
                }
            ).execute()

            logger.info(f"Saved new plan for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving plan/profile: {e}", exc_info=True)

    # ---------------- MAIN CHAT HANDLER ----------------
    async def main_chat_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.model:
            await update.message.reply_text("AI brain not configured. Contact admin.")
            return

        user_id = update.message.from_user.id
        user_message = update.message.text
        logger.info(f"User {user_id}: {user_message}")

        self._log_conversation(user_id, "user", user_message)

        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING,
            )

            chat_session = context.user_data.get("chat_session")

            if not chat_session:
                chat_session = self.model.start_chat()
                context.user_data["chat_session"] = chat_session

                user_data = self._load_profile(user_id)

                if user_data:
                    profile = user_data.get("profile", {})
                    last_plan = user_data.get("last_plan", "No previous plan found.")
                    name = profile.get("name", "friend")

                    initial_prompt = (
                        f"SYSTEM_NOTE: Returning user. Profile: {json.dumps(profile)}. "
                        f"Last plan: '{last_plan}'. Greet them by name ({name}) and ask what they need. "
                        f"Their first message: '{user_message}'."
                        "\n\nREMINDER: If you generate a new wellness PLAN, you MUST follow RULE A and "
                        "include [USER_DATA_JSON] on the first line and [END_OF_PLAN] as the last token."
                    )
                else:
                    initial_prompt = (
                        "SYSTEM_NOTE: Brand new user. Start the health profile questions. "
                        f"Their first message: '{user_message}'."
                        "\n\nREMINDER: When you later generate a full wellness PLAN, you MUST follow RULE A and "
                        "include [USER_DATA_JSON] on the first line and [END_OF_PLAN] as the last token."
                    )

                response = await chat_session.send_message_async(
                    user_message + "\n\nREMINDER: If generating a full wellness plan, you MUST follow RULE A: "
                                   "Start with [USER_DATA_JSON] {...} then 10 numbered points, then [END_OF_PLAN]."
                )

            else:
                # Always remind the model in case this message is asking for a PLAN
                reminder = (
                    "\n\nREMINDER: If the user is asking for a new or updated wellness PLAN, "
                    "you MUST follow RULE A and include [USER_DATA_JSON] on the first line and "
                    "[END_OF_PLAN] as the last token of the message."
                )
                response = await chat_session.send_message_async(user_message + reminder)

            ai_message = response.text
            self._log_conversation(user_id, "ai", ai_message)

            if "[END_OF_PLAN]" in ai_message:
                profile_json = self._extract_profile_json(ai_message)
                if not profile_json:
                    clean = ai_message.replace("[END_OF_PLAN]", "").strip()
                    await update.message.reply_text(clean)
                    return

                plan_text = self._extract_plan_text(ai_message)

                self.save_new_plan_and_profile(user_id, profile_json, plan_text)

                await update.message.reply_text(plan_text.strip())
            else:
                await update.message.reply_text(ai_message)

        except Exception as e:
            logger.error(f"Error in main_chat_handler: {e}", exc_info=True)
            await update.message.reply_text(
                "Sorry, I had a problem. Please try again in a moment."
            )

    # ---------------- RESET COMMAND ----------------
    async def reset_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data.clear()
        await update.message.reply_text(
            "I've cleared our conversation. We can start fresh!"
        )

    # ---------------- RUN BOT ----------------
    def run(self):
        if not self.model:
            logger.error("Cannot run bot: Gemini model not initialised.")
            return

        application = Application.builder().token(self.telegram_token).build()

        application.add_handler(CommandHandler("start", self.main_chat_handler))
        application.add_handler(CommandHandler("reset", self.reset_chat))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.main_chat_handler)
        )

        logger.info("HealthCareBot is running...")
        application.run_polling()


if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        logging.error("Missing TELEGRAM_TOKEN or GEMINI_API_KEY in .env")
    else:
        logging.info("All tokens and keys loaded from .env.")
        bot = HealthCareBot(TELEGRAM_TOKEN, GEMINI_API_KEY)
        bot.run()

