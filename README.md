# BITI MEPhI Books Semantic Search

<img width="920" height="647" alt="output-onlinepngtools (1)" src="https://github.com/user-attachments/assets/de65d40e-eb45-4a4d-ac65-fdb9a296cda6" />

This repository implements an end to end semantic search pipeline for a university book collection. Textual fields consisting of title plus annotation are preprocessed and embedded with a Russian scientific BERT model. The resulting dense vectors are normalised and stored to allow fast approximate retrieval.

## Embeddings

Embeddings are produced by mean pooling token representations from the ruSciBERT model and applying L2 normalisation to each vector. All embeddings are computed in batches and cached to disk to avoid recomputation. The [dataset](https://www.kaggle.com/datasets/alexzyukov/biti-mephi-2k-books-library/) used for this work is a publicly available BITI MEPhI catalogue.

## Clustering and Indexing

To reduce the number of comparisons at query time the embedding set is clustered with K-Means. Cluster count is chosen after simple analysis and validation. Cluster centroids are saved and used to shortlist candidate clusters for each incoming query. At search time a query is embedded with the same model and normalisation, similarities to centroids are computed with cosine similarity, the best clusters are inspected and the most similar book embeddings inside those clusters are returned as matches.

## Processed data

- `data/embeddings.npy` — normalized embeddings for all books.  
- `data/ids.npy` — integer ids aligned with embeddings.  
- `data/df_metadata.csv` — titles, authors, annotations and other metadata.  
- `data/labels.npy`, `data/centroids.npy`, `data/centroids_pca.npy` — clustering artifacts.  

## Telegram bot

A simple Telegram bot demonstrates how to serve the search. The bot embeds the user query, picks top clusters by centroid similarity, searches inside those clusters, formats metadata fields title author year publisher link and annotation and replies with three best matches. The cosine similarity calculation and the search system are also implemented in `bot.py`.

<img width="823" height="621" alt="image" src="https://github.com/user-attachments/assets/a6a348a0-ff1d-4b9b-a631-ffaa0d9dabe0" />
