# Recommender Systems

Systems that predict user preferences and suggest items (products, movies, music, content) based on user behavior and item characteristics.

## Overview

Recommender systems are the algorithms behind:
- Netflix movie suggestions
- Amazon product recommendations
- Spotify playlists
- YouTube video suggestions
- LinkedIn connection recommendations

**Goal:** Predict how much a user will like an item they haven't interacted with yet.

## Types of Recommender Systems

### 1. Collaborative Filtering
**Idea:** "Users who agreed in the past will agree in the future"

**User-Based:** Find similar users, recommend what they liked
**Item-Based:** Find similar items to what the user liked

**Pros:**
- No domain knowledge needed
- Works for any type of item
- Discovers unexpected patterns

**Cons:**
- Cold start problem (new users/items)
- Sparsity issues (most users rate few items)
- Scalability challenges

### 2. Content-Based Filtering
**Idea:** "Recommend items similar to what the user liked before"

Uses item features (genre, director, keywords) to find similar items

**Pros:**
- No cold start for new users
- Can explain recommendations
- No need for other users' data

**Cons:**
- Requires feature engineering
- Limited to user's existing preferences
- Doesn't discover new interests

### 3. Hybrid Systems
**Idea:** Combine multiple approaches for better recommendations

**Pros:**
- Best of both worlds
- Handles cold start better
- More accurate overall

**Cons:**
- More complex to implement
- Harder to debug

## Quick Examples

### Collaborative Filtering (User-Based)
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# User-item rating matrix
ratings = pd.DataFrame({
    'Movie A': [5, 3, 0, 1],
    'Movie B': [3, 1, 4, 3],
    'Movie C': [4, 0, 4, 5],
    'Movie D': [0, 1, 5, 4]
}, index=['User 1', 'User 2', 'User 3', 'User 4'])

# Calculate user similarity
user_similarity = cosine_similarity(ratings.fillna(0))
user_sim_df = pd.DataFrame(user_similarity, 
                            index=ratings.index, 
                            columns=ratings.index)

print(user_sim_df)

# Predict rating for User 1 on Movie D
def predict_rating(ratings, user_sim, user, item):
    similar_users = user_sim[user].drop(user).sort_values(ascending=False)
    similar_ratings = ratings.loc[similar_users.index, item]
    
    # Remove NaN ratings
    mask = ~similar_ratings.isna()
    similar_users = similar_users[mask]
    similar_ratings = similar_ratings[mask]
    
    if len(similar_ratings) == 0:
        return ratings[item].mean()
    
    # Weighted average
    weighted_sum = (similar_users * similar_ratings).sum()
    similarity_sum = similar_users.sum()
    
    return weighted_sum / similarity_sum if similarity_sum > 0 else ratings[item].mean()

predicted = predict_rating(ratings, user_sim_df, 'User 1', 'Movie D')
print(f"Predicted rating: {predicted:.2f}")
```

### Item-Based Collaborative Filtering
```python
# Calculate item similarity
item_similarity = cosine_similarity(ratings.fillna(0).T)
item_sim_df = pd.DataFrame(item_similarity,
                           index=ratings.columns,
                           columns=ratings.columns)

# Recommend items similar to those user liked
def recommend_items(ratings, item_sim, user, n_recommendations=3):
    user_ratings = ratings.loc[user]
    liked_items = user_ratings[user_ratings > 3].index
    
    # Get similar items
    similar_items = {}
    for item in liked_items:
        similar = item_sim_df[item].drop(item).sort_values(ascending=False)
        for sim_item, score in similar.items():
            if pd.isna(user_ratings[sim_item]):
                similar_items[sim_item] = similar_items.get(sim_item, 0) + score
    
    # Sort and return top N
    recommendations = sorted(similar_items.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:n_recommendations]
    return [item for item, score in recommendations]

recommendations = recommend_items(ratings, item_sim_df, 'User 1', n_recommendations=2)
print(f"Recommended items: {recommendations}")
```

## Matrix Factorization

Decompose user-item matrix into user and item latent factors.

### SVD (Singular Value Decomposition)
```python
from scipy.sparse.linalg import svds
import numpy as np

# Fill missing values with 0 or mean
ratings_filled = ratings.fillna(ratings.mean())

# Perform SVD
U, sigma, Vt = svds(ratings_filled, k=2)  # k = number of latent factors

# Reconstruct ratings matrix
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(predicted_ratings, 
                            index=ratings.index,
                            columns=ratings.columns)

print("Predicted ratings:")
print(predicted_df)
```

### Using Surprise Library
```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Prepare data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)

# Train SVD model
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train on full dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict
prediction = algo.predict(uid='user123', iid='item456')
print(f"Predicted rating: {prediction.est}")
```

## Advanced: Deep Learning Approaches

### Neural Collaborative Filtering
```python
import tensorflow as tf
from tensorflow import keras

# Build neural CF model
def build_ncf_model(n_users, n_items, embedding_dim=50):
    # User input
    user_input = keras.Input(shape=(1,), name='user_input')
    user_embedding = keras.layers.Embedding(n_users, embedding_dim)(user_input)
    user_vec = keras.layers.Flatten()(user_embedding)
    
    # Item input
    item_input = keras.Input(shape=(1,), name='item_input')
    item_embedding = keras.layers.Embedding(n_items, embedding_dim)(item_input)
    item_vec = keras.layers.Flatten()(item_embedding)
    
    # Concatenate and add dense layers
    concat = keras.layers.Concatenate()([user_vec, item_vec])
    dense1 = keras.layers.Dense(128, activation='relu')(concat)
    dropout1 = keras.layers.Dropout(0.3)(dense1)
    dense2 = keras.layers.Dense(64, activation='relu')(dropout1)
    dropout2 = keras.layers.Dropout(0.3)(dense2)
    output = keras.layers.Dense(1, activation='linear')(dropout2)
    
    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Train model
model = build_ncf_model(n_users=1000, n_items=500)
model.fit(
    [train_users, train_items], 
    train_ratings,
    validation_split=0.2,
    epochs=10,
    batch_size=256
)

# Predict
predicted_rating = model.predict([user_id, item_id])
```

## Content-Based Filtering

### TF-IDF Similarity
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Item descriptions/features
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Titanic', 'The Notebook'],
    'genres': ['action sci-fi', 'action sci-fi thriller', 'romance drama', 'romance drama']
})

# Calculate TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend similar items
def content_based_recommendations(title, movies, cosine_sim, n_recommendations=3):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

recommendations = content_based_recommendations('The Matrix', movies, cosine_sim)
print(f"Recommended: {recommendations}")
```

## Evaluation Metrics

### Rating Prediction Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.3f}")

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.3f}")
```

**Lower is better** for both RMSE and MAE

### Ranking Metrics
```python
def precision_at_k(recommended, relevant, k):
    """Precision@K: proportion of recommended items that are relevant"""
    recommended_k = recommended[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant))
    return relevant_recommended / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    """Recall@K: proportion of relevant items that are recommended"""
    recommended_k = recommended[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant))
    return relevant_recommended / len(relevant) if len(relevant) > 0 else 0

def average_precision(recommended, relevant):
    """Average Precision: precision averaged over all relevant positions"""
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended):
        if item in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / len(relevant) if len(relevant) > 0 else 0

def mean_average_precision(all_recommended, all_relevant):
    """MAP: mean of average precision across all users"""
    return np.mean([average_precision(rec, rel) 
                    for rec, rel in zip(all_recommended, all_relevant)])

# Example usage
recommended = ['item1', 'item3', 'item5', 'item7', 'item2']
relevant = ['item1', 'item2', 'item4']

print(f"Precision@3: {precision_at_k(recommended, relevant, 3):.3f}")
print(f"Recall@3: {recall_at_k(recommended, relevant, 3):.3f}")
print(f"Average Precision: {average_precision(recommended, relevant):.3f}")
```

### Diversity & Coverage
```python
def diversity(recommendations):
    """Measure how diverse recommendations are"""
    from itertools import combinations
    
    if len(recommendations) < 2:
        return 0
    
    # Calculate average pairwise distance
    pairs = list(combinations(recommendations, 2))
    similarities = [item_similarity(i1, i2) for i1, i2 in pairs]
    return 1 - np.mean(similarities)

def coverage(recommendations, all_items):
    """Percentage of items ever recommended"""
    unique_recommended = set()
    for rec in recommendations:
        unique_recommended.update(rec)
    return len(unique_recommended) / len(all_items)
```

## Handling Cold Start Problem

### New User Cold Start
```python
# 1. Ask for explicit preferences
def onboard_new_user():
    """Ask user to rate popular items"""
    popular_items = get_most_popular_items(n=10)
    # Collect ratings
    # Use these ratings to bootstrap recommendations

# 2. Use demographic information
def demographic_recommendations(age, gender, location):
    """Recommend based on demographic segment"""
    similar_users = find_users_with_demographics(age, gender, location)
    return get_popular_in_segment(similar_users)

# 3. Use content-based initially
def hybrid_cold_start(user_id):
    if is_new_user(user_id):
        # Use content-based or popularity
        return content_based_recommendations(user_id)
    else:
        # Use collaborative filtering
        return collaborative_recommendations(user_id)
```

### New Item Cold Start
```python
# 1. Use item features (content-based)
def recommend_new_item(item_features):
    """Find similar items based on features"""
    similar_items = find_similar_by_features(item_features)
    users_who_liked = get_users_who_liked(similar_items)
    return users_who_liked

# 2. Promote to exploratory users
def promote_new_items(item_id):
    """Show new items to users who explore"""
    exploratory_users = identify_explorers()  # Users with diverse history
    return exploratory_users
```

## Popular Libraries & Tools

### Surprise
```python
from surprise import SVD, KNNBasic, NMF
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

# Load data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

# Try different algorithms
algorithms = {
    'SVD': SVD(),
    'KNN': KNNBasic(),
    'NMF': NMF()
}

for name, algo in algorithms.items():
    results = cross_validate(algo, data, measures=['RMSE'], cv=5)
    print(f"{name} RMSE: {results['test_rmse'].mean():.3f}")
```

### Implicit (for implicit feedback)
```python
import implicit

# For implicit feedback (clicks, views, purchases)
# Not explicit ratings

# Create sparse user-item matrix
from scipy.sparse import csr_matrix
user_item_matrix = csr_matrix((ratings, (users, items)))

# Train ALS model
model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10)
model.fit(user_item_matrix)

# Get recommendations
user_id = 0
recommendations = model.recommend(user_id, user_item_matrix[user_id], N=10)
print(recommendations)
```

### LightFM (hybrid model)
```python
from lightfm import LightFM
from lightfm.data import Dataset

# Build dataset with user and item features
dataset = Dataset()
dataset.fit(users, items, user_features, item_features)

# Build interaction matrix
(interactions, weights) = dataset.build_interactions(interaction_data)

# Build feature matrices
user_features_matrix = dataset.build_user_features(user_features_data)
item_features_matrix = dataset.build_item_features(item_features_data)

# Train hybrid model
model = LightFM(loss='warp')  # WARP: Weighted Approximate-Rank Pairwise
model.fit(interactions, 
          user_features=user_features_matrix,
          item_features=item_features_matrix,
          epochs=10)

# Predict
scores = model.predict(user_ids, item_ids)
```

## Real-World System Architecture

```python
class RecommenderSystem:
    def __init__(self):
        self.cf_model = None  # Collaborative filtering
        self.cb_model = None  # Content-based
        self.popularity_baseline = None
        
    def train(self, interactions, item_features):
        """Train all models"""
        # Train CF
        self.cf_model = train_collaborative_filtering(interactions)
        
        # Train CB
        self.cb_model = train_content_based(item_features)
        
        # Calculate popularity
        self.popularity_baseline = calculate_popularity(interactions)
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate hybrid recommendations"""
        # Check if new user
        if self.is_new_user(user_id):
            return self.cold_start_recommendations(user_id, n_recommendations)
        
        # Get CF recommendations
        cf_recs = self.cf_model.recommend(user_id, n=20)
        
        # Get CB recommendations
        cb_recs = self.cb_model.recommend(user_id, n=20)
        
        # Combine with weights
        combined = self.combine_recommendations(
            cf_recs, cb_recs, 
            cf_weight=0.7, cb_weight=0.3
        )
        
        # Re-rank for diversity
        final_recs = self.diversify(combined, n_recommendations)
        
        return final_recs
    
    def cold_start_recommendations(self, user_id, n):
        """Handle new users"""
        # Use popularity + limited user data
        return self.popularity_baseline[:n]
```

## When to Use Each Approach

### Collaborative Filtering
✅ Large user base with interaction history  
✅ Items are difficult to describe with features  
✅ Want to discover unexpected recommendations  
❌ Cold start problems  
❌ Sparse data  

### Content-Based
✅ Rich item features available  
✅ New users/items (less cold start)  
✅ Need explainable recommendations  
❌ Limited to user's existing preferences  
❌ No serendipity  

### Matrix Factorization
✅ Large-scale systems  
✅ Implicit feedback data  
✅ Good balance of accuracy and speed  
❌ Computationally expensive to train  
❌ Requires regular retraining  

### Deep Learning
✅ Very large datasets  
✅ Complex patterns  
✅ Multiple data sources  
❌ Requires lots of data  
❌ Computationally expensive  
❌ Hard to interpret  

## Best Practices

1. **Start simple** - Begin with popularity baseline, then collaborative filtering
2. **Handle cold start** - Have strategies for new users/items
3. **Diversify recommendations** - Don't just optimize for accuracy
4. **A/B test** - Measure business metrics, not just model metrics
5. **Retrain regularly** - User preferences change over time
6. **Filter inappropriate items** - Business rules matter
7. **Explain recommendations** - "Because you watched X"
8. **Balance exploration vs exploitation** - Show some unexpected items
9. **Monitor biases** - Avoid feedback loops and filter bubbles
10. **Optimize for business goals** - Clicks? Purchases? Engagement?

## Common Pitfalls

❌ **Popularity bias** - Only recommending popular items  
❌ **Filter bubbles** - Never showing new interests  
❌ **Ignoring cold start** - No plan for new users/items  
❌ **Not retraining** - Stale recommendations  
❌ **Overfitting to history** - Users' tastes change  
❌ **Ignoring business constraints** - (inventory, margins, etc.)  
❌ **No diversity** - All recommendations too similar  
❌ **Not A/B testing** - Assuming better metrics = better business results  

## Production Considerations

### Scalability
```python
# Use approximate nearest neighbors for large-scale
from annoy import AnnoyIndex

# Build index
index = AnnoyIndex(embedding_dim, 'angular')
for i, embedding in enumerate(item_embeddings):
    index.add_item(i, embedding)
index.build(10)  # 10 trees

# Fast lookup
similar_items = index.get_nns_by_item(item_id, 10)
```

### Real-Time Recommendations
```python
# Pre-compute recommendations
def precompute_recommendations(all_users):
    """Batch compute recommendations overnight"""
    recommendations_cache = {}
    for user_id in all_users:
        recommendations_cache[user_id] = model.recommend(user_id, n=100)
    return recommendations_cache

# Serve from cache + real-time adjustments
def get_recommendations(user_id, cache, recent_interactions):
    """Serve fast recommendations"""
    # Get pre-computed
    cached_recs = cache.get(user_id, default_recs)
    
    # Filter already interacted
    filtered = [r for r in cached_recs if r not in recent_interactions]
    
    # Maybe add real-time trending items
    final = add_trending_items(filtered[:8]) + filtered[8:10]
    
    return final[:10]
```

### Monitoring
```python
# Track key metrics
metrics = {
    'click_through_rate': clicks / impressions,
    'conversion_rate': purchases / clicks,
    'diversity': calculate_diversity(recommendations),
    'coverage': calculate_coverage(recommendations),
    'novelty': calculate_novelty(recommendations),
    'model_latency': recommendation_time
}
```

---

*Recommender systems are everywhere in modern applications. Start with collaborative filtering, add content-based for cold start, and continuously A/B test to optimize for your business goals!*
