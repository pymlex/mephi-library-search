import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import telebot
import os


with open('config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

TOKEN = cfg['token']
MODEL_NAME = cfg['model']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

df = pd.read_csv('data/df_metadata.csv')
embeddings = np.load('data/embeddings.npy')
embeddings_tensor = torch.from_numpy(embeddings).to(device)
labels = np.load('data/labels.npy')
centroids = np.load('data/centroids.npy')
centroids_tensor = torch.from_numpy(centroids).to(device)

bot = telebot.TeleBot(TOKEN, parse_mode=None)


def embed_text(text):
    inputs = tokenizer(text, 
    	truncation=True, 
    	max_length=512, 
    	padding='longest', 
    	return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        hidden_size = last.size()
        mask = attention_mask.unsqueeze(-1).expand(hidden_size).float()
        summed = (last * mask).sum(1)
        counts = mask.sum(1)
        vector = summed / counts
        vector = F.normalize(vector, p=2, dim=1)
    return vector.squeeze(0)


def find_best_books(query, top_k=3):
    qvec = embed_text(query)

    sims = F.cosine_similarity(qvec.unsqueeze(0), centroids_tensor, dim=1)
    top_clusters = torch.topk(sims, k=3).indices.cpu().numpy()

    idxs = np.concatenate([np.where(labels == c)[0] for c in top_clusters])
    cluster_embs = embeddings_tensor[idxs]

    qvec_unsqueezed = qvec.unsqueeze(0).expand(len(idxs), -1)
    sims2 = F.cosine_similarity(qvec_unsqueezed, cluster_embs, dim=1)

    best_locals = torch.topk(sims2, k=top_k).indices.cpu().numpy()
    chosen_idxs = [int(idxs[i]) for i in best_locals]

    return chosen_idxs


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
        parts.append(f"Год: {int(year)}")
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
    bot.reply_to(message, "Привет. Отправьте команду /search и текст запроса через пробел.")


@bot.message_handler(commands=['search'])
def handle_search(message):
    text = message.text
    query = text.partition(' ')[2].strip()
    if not query:
        bot.reply_to(message, "Использование: /search <текст запроса>")
        return

    chosen_idxs = find_best_books(query, top_k=3)
    replies = []

    for i, idx in enumerate(chosen_idxs, 1):
        row = df.iloc[idx]
        replies.append(f"Результат {i}:\n{format_reply(row)}")

    bot.reply_to(message, "\n\n" + "\n\n".join(replies))


if __name__ == '__main__':
    bot.infinity_polling()
