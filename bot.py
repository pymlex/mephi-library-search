import json
import requests
import pandas as pd
import telebot


with open('config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)


TOKEN = cfg['token']
API_URL = cfg.get('api_url', 'http://localhost:8000')
bot = telebot.TeleBot(TOKEN, parse_mode=None)


def format_reply(row):
    parts = []
    title = row.get('Название')
    if pd.notna(title):
        parts.append(f"Название: {title}")
    authors = row.get('Автор')
    if pd.notna(authors):
        parts.append(f"Автор(ы): {authors}")
    year = row.get('Год издания')
    if pd.notna(year):
        parts.append(f"Год: {year}")
    publisher = row.get('Издательство')
    if pd.notna(publisher):
        parts.append(f"Издательство: {publisher}")
    link = row.get('Сссылка на книгу в ЭБС')
    if pd.notna(link):
        parts.append(f"Ссылка: {link}")
    ann = row.get('Аннотация')
    if pd.notna(ann):
        parts.append(f"Аннотация: {ann}")
    return "\n".join(parts)


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, \
        "Привет! Я бот для поиска по технической "
        "литературе ВИТИ МИФИ. Используйте /search <текст запроса>"
    )


@bot.message_handler(commands=['search'])
def handle_search(message):
    text = message.text
    query = text.partition(' ')[2].strip()

    if not query:
        bot.reply_to(message, "Использование: /search <текст запроса>")
        return
    resp = requests.post(f"{API_URL}/search", json={'query': query, 'top_k': 3})
    
    if resp.status_code != 200:
        bot.reply_to(message, f"Ошибка сервера ({resp.status_code}). Попробуйте позже.")
        return

    data = resp.json()
    latency = resp.headers.get('X-Search-Time')
    if not data:
        bot.reply_to(message, "Ничего не найдено")
        return
    replies = []

    for i, row in enumerate(data, 1):
        replies.append(f"Результат {i}:\n{format_reply(row)}")
        
    footer = f"\n\n(поиск занял {float(latency):.3f}s)" if latency is not None else ""
    bot.reply_to(message, "\n\n".join(replies) + footer)


if __name__ == '__main__':
    bot.infinity_polling()
