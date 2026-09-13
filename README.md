# Personalized Movie Recommendation

A hybrid movie recommendation system combining **collaborative filtering (SVD matrix factorization)** with **content-based filtering** (TMDB metadata), evaluated using standard offline ranking metrics.

## Overview

This project builds a personalized Top-K movie recommendation pipeline by merging two complementary data sources:

- **[MovieLens](https://grouplens.org/datasets/movielens/) (ml-latest-small)** — 100,836 real user ratings across 610 users, used to learn collaborative filtering signal from actual user behavior.
- **[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)** — rich content metadata (cast, crew, genres, keywords, plot overview) used to compute content-based similarity between movies.

The two datasets are linked via MovieLens's `links.csv` mapping (`movieId` → `tmdbId`), allowing every user rating to be paired with rich content features for the corresponding movie.

## Approach

### 1. Content-Based Filtering
- Combines cast, director, genres, and keywords into a single text "soup" per movie
- Vectorized using **TF-IDF**
- Similarity computed via **cosine similarity** between movie vectors

### 2. Collaborative Filtering — SVD Matrix Factorization
- Builds a sparse user-item ratings matrix
- Learns latent user and item factors via **Truncated SVD**
- Generates Top-K candidates by scoring unseen movies against a user's learned latent profile

> An earlier item-based KNN approach (single-seed similarity) was also implemented and benchmarked — SVD outperformed it substantially (see Results).

### 3. Hybrid Re-Ranking
- SVD generates an initial candidate pool
- Candidates are re-ranked using a weighted blend of:
  - Collaborative filtering rank position
  - Content-based similarity to the user's highest-rated movie
- Blend weights were tuned via grid search across `cf_weight` / `content_weight` combinations

## Offline Evaluation

Since no live user-interaction (click) data was available, an **offline evaluation** methodology was used:

1. Ratings were split into train/test sets per user
2. Movies rated **≥ 4.0** in the held-out test set were treated as "relevant"
3. Recommendations were generated using only the training data
4. Standard ranking metrics were computed by comparing recommendations against the held-out relevant set:

| Metric | What it measures |
|---|---|
| **Precision@K** | Of the top K recommendations, how many were relevant |
| **Recall@K** | Of all relevant movies, how many were captured in the top K |
| **NDCG@K** | Precision that also rewards ranking relevant items *higher* in the list |

Results were averaged across multiple random train/test splits to account for variance.

## Results (K=10)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Item-based KNN (single-seed baseline) | 0.055 | 0.064 | 0.204 |
| SVD Matrix Factorization | 0.193 | 0.256 | 0.534 |
| **Hybrid (SVD + Content re-ranking)** | **0.194** | **0.256** | **0.536** |

- Switching from item-based KNN to SVD gave a **~2.6x improvement in NDCG@10**, the single largest gain in the project.
- Content-based re-ranking on top of SVD candidates provided only marginal additional gains (~1% relative NDCG improvement), indicating the collaborative signal alone captures most of the useful ranking structure in this dataset.

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn** — `TfidfVectorizer`, `TruncatedSVD`, `NearestNeighbors`, `train_test_split`
- **SciPy** — sparse matrix operations

## Project Structure

```
├── Recommendation_System.ipynb   # Main notebook: data loading, models, evaluation
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   ├── ratings.csv
│   ├── movies.csv
│   └── links.csv
└── README.md
```

## How to Run

1. Download the [MovieLens ml-latest-small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) dataset and the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Place all CSV files in the `data/` directory (update paths in the notebook if needed)
3. Run `Recommendation_System.ipynb` top to bottom — it will:
   - Merge the datasets via `links.csv`
   - Train the content-based and SVD models
   - Run offline evaluation and print Precision@K / Recall@K / NDCG@K

## Future Work

- Aggregate collaborative candidates across a user's **entire** rating history rather than a single seed movie for content-based scoring
- Experiment with alternative matrix factorization methods (implicit ALS)
- Add a real-time feedback pipeline to incorporate live user interactions (clicks, watch progress, ratings) for continuous personalization
- Extend evaluation to the larger MovieLens 25M dataset to validate results at scale
- Build a type-ahead search feature using BM25 + semantic embeddings for query understanding

## Author

Abhinav Adarsh — [LinkedIn](https://www.linkedin.com/in/abhinav-adarsh-0769b8223/) | [GitHub](https://github.com/AbhiAadi) | [Portfolio](https://abhinav-portfolio-five-sepia.vercel.app/)
