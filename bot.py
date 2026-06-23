import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import joblib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# загружаем модель и кодировщики
model      = joblib.load('lr_model.pkl')
le         = joblib.load('label_encoder.pkl')
le_country = joblib.load('le_country.pkl')

ALL_SKILLS = [
    'python', 'sql', 'data_analysis', 'machine_learning',
    'communication', 'data_visualization', 'aws', 'data_engineering',
    'project_management', 'tableau', 'data_science', 'r',
    'data_modeling', 'data_management', 'java', 'data_warehousing',
    'spark', 'problem_solving', 'data_governance', 'analytical_skills',
    'cloud_computing', 'teamwork', 'hadoop', 'computer_science',
    'snowflake', 'power_bi', 'scala', 'data_mining', 'collaboration',
    'statistics', 'business_intelligence', 'attention_to_detail',
    'agile', 'data_quality', 'azure', 'data_integration', 'etl',
    'bachelors_degree', 'leadership', 'reporting', 'data_pipelines',
    'excel', 'data_architecture', 'nosql', 'kafka',
    'data_security', 'tensorflow'
]

SKILL_GROUPS = {
    '💻 Языки программирования': [
        'python', 'sql', 'r', 'java', 'scala'
    ],
    '☁️ Облако и инфраструктура': [
        'aws', 'azure', 'cloud_computing', 'snowflake',
        'spark', 'hadoop', 'kafka', 'nosql'
    ],
    '📊 Аналитика и визуализация': [
        'tableau', 'power_bi', 'excel', 'data_visualization',
        'reporting', 'statistics', 'data_mining'
    ],
    '🔧 Data Engineering': [
        'etl', 'data_pipelines', 'data_warehousing',
        'data_integration', 'data_architecture', 'data_engineering'
    ],
    '🤖 ML и AI': [
        'machine_learning', 'tensorflow', 'data_science',
        'data_modeling', 'data_analysis', 'computer_science'
    ],
    '🗄️ Управление данными': [
        'data_governance', 'data_quality', 'data_management',
        'data_security', 'business_intelligence'
    ],
    '🤝 Soft skills': [
        'communication', 'problem_solving', 'leadership',
        'project_management', 'teamwork', 'agile',
        'analytical_skills', 'attention_to_detail',
        'collaboration', 'bachelors_degree'
    ]
}

all_grouped = [s for skills in SKILL_GROUPS.values() for s in skills]
missing = [s for s in ALL_SKILLS if s not in all_grouped]
extra   = [s for s in all_grouped if s not in ALL_SKILLS]

print(f"Модель загружена!")
print(f"Классы: {le.classes_}")
print(f"Скилов: {len(ALL_SKILLS)}")
print(f"Групп скилов: {len(SKILL_GROUPS)}")
print(f"Не попали в группы: {missing}")
print(f"Лишние (нет в модели): {extra}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['selected_skills'] = []
    context.user_data['current_group']   = 0
    await update.message.reply_text(
        "👋 Привет! Я *SkillPathBot*\n\n"
        "Помогу определить какое направление в IT\n"
        "подходит тебе по навыкам.\n\n"
        "Отмечай галочками навыки которыми владеешь.\n"
        "Можно пропустить группу если навыков нет.\n\n"
        "Поехали! 🚀",
        parse_mode='Markdown'
    )
    await show_skill_group(update, context)


async def show_skill_group(update, context):
    group_names = list(SKILL_GROUPS.keys())
    current     = context.user_data['current_group']
    group_name  = group_names[current]
    skills      = SKILL_GROUPS[group_name]
    selected    = context.user_data['selected_skills']

    keyboard = []
    row = []
    for skill in skills:
        label = f"✅ {skill}" if skill in selected else skill
        row.append(InlineKeyboardButton(label, callback_data=f"skill_{skill}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    keyboard.append([
        InlineKeyboardButton("Пропустить ⏭", callback_data="skip"),
        InlineKeyboardButton("Далее →",       callback_data="next")
    ])

    text = (
        f"Шаг {current + 1} из {len(SKILL_GROUPS)}\n"
        f"{group_name}\n\n"
        f"Выбрано навыков: {len(selected)}"
    )

    query = update.callback_query if hasattr(update, 'callback_query') else None

    if query:
        try:
            await query.edit_message_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception:
            await query.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    else:
        await update.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    selected = context.user_data.get('selected_skills', [])
    current  = context.user_data.get('current_group', 0)

    if data.startswith('skill_'):
        skill = data.replace('skill_', '')
        if skill in selected:
            selected.remove(skill)
        else:
            selected.append(skill)
        context.user_data['selected_skills'] = selected
        await show_skill_group(update, context)

    elif data in ('next', 'skip'):
        context.user_data['current_group'] = current + 1
        if context.user_data['current_group'] >= len(SKILL_GROUPS):
            await predict_and_show(query, context)
        else:
            await show_skill_group(update, context)

    elif data == 'restart':
        context.user_data['selected_skills'] = []
        context.user_data['current_group']   = 0
        await show_skill_group(update, context)


async def predict_and_show(query, context):
    selected = context.user_data.get('selected_skills', [])

    if len(selected) == 0:
        await query.message.reply_text(
            "⚠️ Ты не выбрал ни одного навыка.\n"
            "Напиши /start чтобы начать заново."
        )
        return

    try:
        X = np.array([[1 if s in selected else 0 for s in ALL_SKILLS]])
        country = np.array([[4]])  # United States по умолчанию
        X = np.hstack([X, country])

        print(f"X shape: {X.shape}")
        print(f"Ожидает признаков: {model.n_features_in_}")

        probabilities = model.predict_proba(X)[0]

        # выводим ВСЕ 4 класса с вероятностями
        all_results = sorted(
            zip(le.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # детализация групп
        group_details = {
            'Engineering':          'Data Engineer, ML Engineer, DevOps, Software Engineer',
            'Analytics':            'Data Analyst, Data Scientist',
            'Operations & Support': 'Database Admin, Data Management, Data Center',
            'Architecture & Design':'Data Architect'
        }

        result = (
            "🎯 Результат анализа навыков\n\n"
            f"📊 Выбрано навыков: {len(selected)}\n\n"
            "Подходящие направления:\n\n"
        )

        medals = ['🥇', '🥈', '🥉', '4️⃣']
        for i, (group, prob) in enumerate(all_results):
            bar = '█' * int(prob * 10)
            detail = group_details.get(group, '')
            result += f"{medals[i]} {group}\n"
            result += f"    {bar} {prob*100:.0f}%\n"
            result += f"    {detail}\n\n"

        result += "─────────────────────\n"
        result += "🔍 Твои навыки:\n"
        result += ', '.join(selected[:10])
        if len(selected) > 10:
            result += f" и ещё {len(selected)-10}..."
        result += "\n\nХочешь попробовать снова?"

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Начать заново", callback_data="restart")
        ]])

        await query.message.reply_text(
            result,
            reply_markup=keyboard
        )

    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        await query.message.reply_text(
            f"⚠️ Ошибка: {str(e)}\n"
            "Напиши /start чтобы начать заново."
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *SkillPathBot — помощник по карьере в IT*\n\n"
        "Команды:\n"
        "/start — начать анализ навыков\n"
        "/help  — справка\n\n"
        "Отмечай навыки галочками и получи\n"
        "рекомендацию по направлению в IT.",
        parse_mode='Markdown'
    )


def main():
    TOKEN = os.getenv('BOT_TOKEN')

    if not TOKEN:
        print("❌ Ошибка: BOT_TOKEN не найден!")
        print("Установи переменную окружения BOT_TOKEN")
        return

    print("🤖 Запускаем SkillPathBot...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help',  help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    print("✅ Бот запущен! Нажми Ctrl+C для остановки.")
    app.run_polling()


if __name__ == '__main__':
    main()
