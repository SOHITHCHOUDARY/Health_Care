# 🧠 HealthMateBot — AI General Health & Wellness Telegram Bot

HealthMateBot is an **AI-powered Telegram bot** that gives **simple daily health & wellness guidance** using **Google Gemini**.  
It remembers user profiles and wellness plans using **Supabase (PostgreSQL)** and provides **personalized suggestions** every time.

> ⚠️ The bot is **not a doctor**. It does **not diagnose diseases** and redirects users to real medical help in emergencies.

---

## 🚀 Features
- 🤝 Onboarding new users (name, age, gender, height, weight, activity level, recurring issues)
- 🔁 Detects returning users and continues from last plan
- 🧾 Generates **personalized 10-point wellness plans**
- 💬 Answers **simple health & lifestyle questions**
- 🗄️ Stores **user profiles & plan history** in Supabase
- 💡 Built-in safety rules for emergency symptoms
- 🔄 `/reset` command to restart setup
- 🧩 OOP-based bot structure for easy maintenance

---

## 🛠 Tech Stack
| Component | Technology |
|----------|------------|
| AI Engine | Google Gemini (gemini-2.5-flash) |
| Backend | Python |
| Framework | python-telegram-bot |
| Database | Supabase (PostgreSQL) |
| Config | dotenv |

---

