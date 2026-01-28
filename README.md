# BITI MEPhI Books Semantic Search
<p align="center">
  <img width="583" height="332" alt="2k books" src="https://github.com/user-attachments/assets/c8ebea26-fef3-43f3-8dab-d7f44c2cdfc0" />
</p>
This project presents a practical semantic search system for the BITI MEPhI university library, aimed at retrieving relevant scientific literature from natural language queries. It relies on dense text representations produced by a Russian scientific BERT model, combines them with unsupervised clustering to reduce search complexity. The developed search system is wrapped in a Telegram bot.

## Embeddings

Embeddings are produced by mean pooling token representations from the ruSciBERT model and applying L2 normalisation to each vector. All embeddings are computed in batches and cached to disk to avoid recomputation. The 2k books [dataset](https://www.kaggle.com/datasets/alexzyukov/biti-mephi-2k-books-library/) used for this work is a publicly available BITI MEPhI catalogue.

<img width="783" height="324" alt="image" src="https://github.com/user-attachments/assets/84d7e534-d94f-4705-9dfd-d7dd265064e2" />

To reduce dimensionality and memory footprint, a Principal Component Analysis (PCA) model is trained on the original 768-dimensional embeddings and used to compress them to 227 dimensions while preserving approximately 95% of the variance. The fitted PCA model is saved to disk and reused at inference time. All book embeddings are stored already in the reduced space, and each incoming query embedding is projected with the same PCA transformation before similarity computation.

## Clustering and Indexing

To reduce the number of comparisons at query time the embedding set is clustered with K-Means. Cluster count is chosen after simple analysis and validation. Cluster centroids are saved and used to shortlist candidate clusters for each incoming query. At search time a query is embedded with the same model and normalisation, similarities to centroids are computed with cosine similarity, the best clusters are inspected and the most similar book embeddings inside those clusters are returned as matches.

## Processed data

- `data/embeddings.npy` — normalized embeddings for all books.  
- `data/ids.npy` — integer ids aligned with embeddings.  
- `data/df_metadata.csv` — titles, authors, annotations and other metadata.  
- `data/labels.npy`, `data/centroids.npy`, `data/centroids_pca.npy` — clustering artifacts.  

## Inference

A simple Telegram bot demonstrates how to serve the search. The core search logic is exposed through a FastAPI service, which loads the trained model, PCA transformation and clustering artifacts, embeds incoming queries and returns matching book metadata via a REST endpoint.

The bot sends user queries to this FastAPI backend, receives search results, formats metadata fields title author year publisher link and annotation and replies with three best matches. The cosine similarity calculation and the search system are also implemented in bot.py.

<img width="411" height="310" alt="telegram bot" src="https://github.com/user-attachments/assets/a6a348a0-ff1d-4b9b-a631-ffaa0d9dabe0" />

Processing time of a single query on an NVIDIA T4 GPU in Google Colab is approximately 15 ms with a standard deviation of 2 ms.
