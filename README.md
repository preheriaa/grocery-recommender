# Grocery Recommender System — Pattern Mining & Collaborative Filtering

A hybrid recommendation engine built on grocery transaction data, combining association rule mining (Apriori) with user-based collaborative filtering to generate personalised product recommendations.

## Problem Statement

Retail businesses rely on recommender systems to increase basket size and improve customer experience. This project builds and evaluates two recommendation approaches — frequent pattern mining and collaborative filtering — then combines them into a hybrid system that falls back gracefully when one method has insufficient data.

## Dataset

- **Training set:** `Groceries data train.csv` — historical user-item transaction records
- **Test set:** `Groceries data test.csv` — held-out transactions for evaluation
- **Features:** `user_id`, item descriptions, purchase history

## Project Structure

```
├── MBD_2025_Tri1_assignment3_a1909636_prashant_shrestha_version06.ipynb  # Main notebook
└── Groceries data train.csv / Groceries data test.csv                    # Dataset
```

## Methodology

### 1. Data Preprocessing
- Removed null rows from training data
- Transformed transaction data into a binary user-item matrix suitable for Apriori

### 2. Frequent Pattern Mining (Apriori)
- Applied the Apriori algorithm to identify frequent itemsets (`min_support = 0.005`)
- Generated association rules filtered by `min_confidence = 0.4` and `lift > 1`
- Built an item-association dictionary mapping each product to its frequently co-purchased items
- Used these associations to generate top-5 product recommendations per user

### 3. User-Based Collaborative Filtering
- Computed user-user cosine similarity from the purchase matrix
- For a given user, identified most similar users and aggregated their item preferences (weighted by similarity)
- Excluded items the user had already purchased, returning top-5 novel recommendations

### 4. Hybrid Recommender
- Primary: Frequent pattern mining recommendations
- Fallback: If fewer than 5 pattern-based recommendations found, supplements with collaborative filtering results
- Ensures every user receives at least 5 recommendations regardless of purchase history depth

### 5. Evaluation
- Evaluated on the held-out test set using:
  - **Precision** — what fraction of recommended items were relevant
  - **Recall** — what fraction of relevant items were recommended
  - **F1 Score** — harmonic mean of precision and recall

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| pandas, NumPy | Data manipulation |
| mlxtend | Apriori algorithm, association rules |
| scikit-learn | Cosine similarity, evaluation metrics |
| Matplotlib, Seaborn | Visualisation |

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

# Open the notebook
jupyter notebook MBD_2025_Tri1_assignment3_a1909636_prashant_shrestha_version06.ipynb
```

> Update file paths to point to your local copies of the training and test CSV files.

## Key Concepts Demonstrated

- Apriori algorithm and association rule mining
- Collaborative filtering with cosine similarity
- Hybrid recommender design with fallback logic
- Precision/Recall/F1 evaluation for recommendation quality

## Author

**Prashant Shrestha**  
Master of Science in Data Science — University of Adelaide (2025)  
[LinkedIn](https://linkedin.com/in/preheria)
