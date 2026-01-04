# BITI MEPhI Books Semantic Search

<img width="490" height="300" alt="2k books" src="https://github.com/user-attachments/assets/58385e9c-8fca-4b02-b50d-4214f14f9db3" />

This project presents a practical semantic search system for the BITI MEPhI university library, aimed at retrieving relevant scientific literature from natural language queries. It relies on dense text representations produced by a Russian scientific BERT model, combines them with unsupervised clustering to reduce search complexity. The developed search system is wrapped in a Telegram bot.

## Embeddings

Embeddings are produced by mean pooling token representations from the ruSciBERT model and applying L2 normalisation to each vector. All embeddings are computed in batches and cached to disk to avoid recomputation. The 2k books [dataset](https://www.kaggle.com/datasets/alexzyukov/biti-mephi-2k-books-library/) used for this work is a publicly available BITI MEPhI catalogue.

## Clustering and Indexing

To reduce the number of comparisons at query time the embedding set is clustered with K-Means. Cluster count is chosen after simple analysis and validation. Cluster centroids are saved and used to shortlist candidate clusters for each incoming query. At search time a query is embedded with the same model and normalisation, similarities to centroids are computed with cosine similarity, the best clusters are inspected and the most similar book embeddings inside those clusters are returned as matches.

## Processed data

- `data/embeddings.npy` — normalized embeddings for all books.  
- `data/ids.npy` — integer ids aligned with embeddings.  
- `data/df_metadata.csv` — titles, authors, annotations and other metadata.  
- `data/labels.npy`, `data/centroids.npy`, `data/centroids_pca.npy` — clustering artifacts.  

## Telegram bot

A simple Telegram bot demonstrates how to serve the search. The bot embeds the user query, picks top clusters by centroid similarity, searches inside those clusters, formats metadata fields title author year publisher link and annotation and replies with three best matches. The cosine similarity calculation and the search system are also implemented in `bot.py`.

<img width="411" height="310" alt="telegram bot" src="https://github.com/user-attachments/assets/a6a348a0-ff1d-4b9b-a631-ffaa0d9dabe0" />
