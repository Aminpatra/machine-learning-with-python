# K-Means Clustering

An unsupervised learning algorithm that partitions data into K distinct, non-overlapping clusters based on feature similarity.

## How It Works

K-Means groups data points by iteratively assigning them to the nearest cluster center:

1. **Initialize** - Randomly place K centroids in the feature space
2. **Assign** - Assign each point to the nearest centroid (cluster)
3. **Update** - Recalculate centroids as the mean of assigned points
4. **Repeat** - Continue steps 2-3 until convergence (centroids stop moving)

**Goal:** Minimize within-cluster variance (inertia/distortion)

## Key Characteristics

**Strengths:**
- Simple and easy to understand
- Fast and efficient (O(n·K·i·d)) where i = iterations, d = dimensions
- Scales well to large datasets
- Works well with spherical/globular clusters
- Guaranteed to converge
- Easy to implement and interpret

**Weaknesses:**
- Must specify K (number of clusters) in advance
- Sensitive to initial centroid placement
- Assumes clusters are spherical and similar size
- Sensitive to outliers
- Struggles with non-convex shapes
- Only works with numerical data

## Quick Example

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create and train
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Predict cluster for new data
new_labels = kmeans.predict(X_new)

# Visualize (2D data)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], 
            marker='X', s=200, c='red', edgecolors='black')
plt.show()
```

## The Algorithm in Detail

### Initialization Methods

**K-Means++ (Default - Recommended):**
```python
kmeans = KMeans(init='k-means++', n_clusters=3)
```
- Smart initialization: spreads initial centroids apart
- Faster convergence, better results
- **Always use this** unless you have a good reason not to

**Random Initialization:**
```python
kmeans = KMeans(init='random', n_clusters=3)
```
- Randomly selects K points as initial centroids
- Can lead to poor local minima
- Faster but less reliable

**Manual Initialization:**
```python
initial_centroids = np.array([[1, 2], [3, 4], [5, 6]])
kmeans = KMeans(init=initial_centroids, n_clusters=3)
```
- Provide your own starting centroids
- Use when you have domain knowledge

### Distance Metric

K-Means uses **Euclidean distance** by default:
```
d(p, q) = √[(p₁-q₁)² + (p₂-q₂)² + ... + (pₙ-qₙ)²]
```

**Important:** Feature scaling is critical because distance-based!

## Choosing K (Number of Clusters)

### Elbow Method
```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method')
plt.show()
```

**Look for:** The "elbow" point where inertia decreases slowly after a sharp drop

### Silhouette Score
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot
plt.plot(K_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()
```

**Range:** -1 to 1
- **Close to 1:** Well-separated clusters
- **Close to 0:** Overlapping clusters
- **Negative:** Points assigned to wrong clusters

### Gap Statistic
```python
from sklearn.cluster import KMeans
import numpy as np

def gap_statistic(X, max_k=10, n_refs=10):
    gaps = []
    for k in range(1, max_k + 1):
        # Cluster actual data
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        actual_inertia = kmeans.inertia_
        
        # Cluster random reference data
        ref_inertias = []
        for _ in range(n_refs):
            random_data = np.random.uniform(X.min(), X.max(), X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42)
            kmeans_ref.fit(random_data)
            ref_inertias.append(kmeans_ref.inertia_)
        
        # Calculate gap
        gap = np.log(np.mean(ref_inertias)) - np.log(actual_inertia)
        gaps.append(gap)
    
    return gaps

gaps = gap_statistic(X, max_k=10)
optimal_k = np.argmax(gaps) + 1
```

**Choose K where:** Gap statistic is largest

## Key Parameters

```python
KMeans(
    n_clusters=8,              # Number of clusters (MUST SET)
    init='k-means++',          # Initialization method
    n_init=10,                 # Number of times to run with different seeds
    max_iter=300,              # Maximum iterations per run
    tol=1e-4,                  # Convergence tolerance
    random_state=42,           # Reproducibility
    algorithm='lloyd'          # 'lloyd' or 'elkan' (faster for well-separated)
)
```

### n_init Parameter
```python
# Run 10 times with different initializations, keep best
kmeans = KMeans(n_clusters=3, n_init=10)
```
- Multiple runs help avoid poor local minima
- Higher = more reliable but slower
- Default: 10 (good balance)

## Evaluation Metrics

### Inertia (Within-Cluster Sum of Squares)
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(f"Inertia: {kmeans.inertia_}")
```
- Sum of squared distances to nearest centroid
- Lower is better
- Use for comparing same dataset with different K

### Silhouette Score
```python
from sklearn.metrics import silhouette_score

labels = kmeans.labels_
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```
- Measures how similar points are to their own cluster vs. other clusters
- Range: [-1, 1], higher is better

### Davies-Bouldin Index
```python
from sklearn.metrics import davies_bouldin_score

labels = kmeans.labels_
score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Score: {score:.3f}")
```
- Ratio of within-cluster to between-cluster distances
- Lower is better (0 is perfect)

### Calinski-Harabasz Index
```python
from sklearn.metrics import calinski_harabasz_score

labels = kmeans.labels_
score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {score:.3f}")
```
- Ratio of between-cluster to within-cluster dispersion
- Higher is better

## Feature Scaling - CRITICAL!

**K-Means is distance-based, so scaling is essential!**

```python
from sklearn.preprocessing import StandardScaler

# Always scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Then apply K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Transform centroids back to original scale (optional)
original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
```

**Why?** Features with larger scales dominate distance calculations.

## Advanced Techniques

### Mini-Batch K-Means (For Large Datasets)
```python
from sklearn.cluster import MiniBatchKMeans

# Much faster for large datasets
mb_kmeans = MiniBatchKMeans(
    n_clusters=3,
    batch_size=100,        # Process in batches
    max_iter=100,
    random_state=42
)
mb_kmeans.fit(X)
```

**Benefits:**
- 10-100x faster than standard K-Means
- Lower memory usage
- Slight trade-off in quality

**Use when:** Dataset > 10,000 samples

### Handling Categorical Features
```python
from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X_categorical)

# Combine with numerical features
X_combined = np.hstack([X_numerical, X_encoded])

# Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
```

**Alternative:** Use K-Modes for categorical data

## Visualization

### 2D Visualization
```python
import matplotlib.pyplot as plt

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                      c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], 
            marker='X', s=300, c='red', edgecolors='black', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.colorbar(scatter, label='Cluster')
plt.show()
```

### 3D Visualization
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
           c=labels, cmap='viridis')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
           marker='X', s=300, c='red', edgecolors='black')
plt.show()
```

### High-Dimensional Data (PCA)
```python
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Cluster in original space
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_scaled)

# Visualize in 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('K-Means Clustering (PCA Projection)')
plt.show()
```

## Common Applications

### Customer Segmentation
```python
# Segment customers by purchasing behavior
features = ['total_spend', 'frequency', 'recency', 'avg_order_value']
X = customer_data[features]

# Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4)
customer_data['segment'] = kmeans.fit_predict(X_scaled)

# Analyze segments
segment_summary = customer_data.groupby('segment')[features].mean()
```

### Image Compression
```python
from sklearn.cluster import KMeans
import numpy as np

# Load image as array (pixels × RGB)
image = plt.imread('image.jpg')
pixels = image.reshape(-1, 3)

# Cluster colors
kmeans = KMeans(n_clusters=16)  # Reduce to 16 colors
kmeans.fit(pixels)

# Replace each pixel with its cluster center
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = compressed_pixels.reshape(image.shape)
```

### Anomaly Detection
```python
# Outliers are far from all centroids
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Calculate distance to nearest centroid
distances = np.min(kmeans.transform(X_scaled), axis=1)

# Identify anomalies (e.g., top 5%)
threshold = np.percentile(distances, 95)
anomalies = distances > threshold
```

## When to Use

✅ **Exploratory data analysis** - Discover natural groupings  
✅ **Customer segmentation** - Group similar customers  
✅ **Image compression** - Reduce color palette  
✅ **Document clustering** - Group similar documents  
✅ **Anomaly detection** - Find outliers  
✅ **Feature engineering** - Create cluster-based features  
✅ **Large datasets** - Fast and scalable  

❌ **Non-spherical clusters** (use DBSCAN or Gaussian Mixture)  
❌ **Varying cluster sizes** (use hierarchical clustering)  
❌ **Different cluster densities** (use DBSCAN)  
❌ **Categorical data** (use K-Modes instead)  
❌ **Need hierarchical structure** (use Agglomerative Clustering)  

## K-Means vs Other Clustering Algorithms

| Aspect | K-Means | DBSCAN | Hierarchical | Gaussian Mixture |
|--------|---------|--------|--------------|------------------|
| **Speed** | Fast | Medium | Slow | Medium |
| **Scalability** | Excellent | Good | Poor | Medium |
| **Cluster shape** | Spherical only | Any | Any | Elliptical |
| **Outliers** | Sensitive | Robust | Sensitive | Robust |
| **K selection** | Must specify | Auto | Auto | Must specify |
| **Use case** | General purpose | Spatial, density | Hierarchical | Probabilistic |

## Best Practices

1. **Always scale features** - Use StandardScaler
2. **Run multiple times** - Use n_init=10 or more
3. **Try different K values** - Use elbow method and silhouette score
4. **Check cluster sizes** - Very uneven = bad fit
5. **Visualize results** - Use PCA for high dimensions
6. **Remove outliers** - Or use robust alternatives
7. **Domain knowledge** - Use it to validate cluster count
8. **Evaluate quality** - Don't just trust inertia

## Common Pitfalls

❌ **Not scaling features** - Dominated by large-scale features  
❌ **Using random initialization** - Use k-means++ instead  
❌ **Picking wrong K** - Validate with multiple methods  
❌ **Assuming spherical clusters** - Wrong algorithm choice  
❌ **Ignoring outliers** - Can skew centroids significantly  
❌ **No validation** - Always check clustering quality  

## Complete Example Workflow

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Find optimal K (Elbow Method)
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot both metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
plt.show()

# 3. Train final model with optimal K (e.g., K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 4. Evaluate
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")

# 5. Analyze clusters
for i in range(optimal_k):
    cluster_data = X[labels == i]
    print(f"\nCluster {i}: {len(cluster_data)} samples")
    print(cluster_data.mean())

# 6. Visualize (if 2D or use PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.colorbar(scatter, label='Cluster')
plt.show()
```

---

*K-Means is the go-to clustering algorithm for most unsupervised learning tasks. Master the elbow method and silhouette analysis to find optimal cluster counts!*
