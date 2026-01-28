# models.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from joblib import load
from sklearn.preprocessing import normalize


with open('config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

MODEL_NAME = cfg['model']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

pca = load('data/pca_reduce.pkl')
df = pd.read_csv('data/df_metadata.csv')
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')
centroids = np.load('data/centroids.npy')

embeddings_norm = normalize(embeddings, axis=1)
centroids_norm = normalize(centroids, axis=1)


def embed_text(text):
    inputs = tokenizer(text, truncation=True, max_length=512, padding='longest', return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last.size()).float()
        summed = (last * mask).sum(1)
        counts = mask.sum(1)
        vector = summed / counts
        vector = F.normalize(vector, p=2, dim=1)
    return vector.cpu().numpy().reshape(-1)


def find_best_books(query, top_k=3, top_clusters_k=3):
    qvec = embed_text(query)
    q_pca = pca.transform(qvec.reshape(1, -1))[0]
    q_norm = q_pca / np.linalg.norm(q_pca)
    sims_cent = centroids_norm.dot(q_norm)
    top_clusters = sims_cent.argsort()[::-1][:top_clusters_k]
    idxs_list = [np.where(labels == c)[0] for c in top_clusters]
    idxs = np.concatenate(idxs_list)
    sub_embs = embeddings_norm[idxs]
    sims = sub_embs.dot(q_norm)
    top_local = sims.argsort()[::-1][:top_k]
    chosen = [int(idxs[i]) for i in top_local]
    return chosen
