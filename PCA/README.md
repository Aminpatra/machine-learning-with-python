# Principal Component Analysis (PCA)

A dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance.

## How It Works

PCA finds new axes (principal components) that capture the most variation in the data:

1. **Standardize data** - Center and scale features (mean=0, std=1)
2. **Compute covariance matrix** - Measure relationships between features
3. **Calculate eigenvectors & eigenvalues** - Find directions of maximum variance
4. **Select components** - Keep top K components with largest eigenvalues
5. **Transform data** - Project original data onto new component axes

**Key Insight:** Transform correlated features into uncorrelated principal components, ordered by importance.

## Key Characteristics

**Strengths:**
- Reduces dimensionality while preserving information
- Removes multicollinearity (creates orthogonal features)
- Speeds up machine learning algorithms
- Reduces overfitting (fewer features)
- Enables visualization of high-dimensional data
- No hyperparameters to tune (just choose # components)

**Weaknesses:**
- Linear transformation only (cannot capture non-linear relationships)
- Principal components are harder to interpret
- Assumes high variance = high importance
- Sensitive to feature scaling
- Information loss with dimensionality reduction
- Not suitable for categorical data

## Quick Example

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Always scale first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# 3. Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Access components
print(f"Component shape: {pca.components_.shape}")
```

## Mathematical Foundation

### What Are Principal Components?

**Principal Component (PC):** A new axis in the feature space
- PC1: Direction of maximum variance
- PC2: Direction of maximum remaining variance (orthogonal to PC1)
- PC3: Direction of maximum remaining variance (orthogonal to PC1 & PC2)
- And so on...

**Mathematically:**
```
PC_i = w_i1·x_1 + w_i2·x_2 + ... + w_in·x_n
```
Where w_ij are the loadings (weights) and x_j are the original features.

### Eigenvalues & Eigenvectors

**Eigenvectors:** Directions of principal components (the new axes)  
**Eigenvalues:** Amount of variance explained by each component

```python
# Eigenvalues (variance explained)
eigenvalues = pca.explained_variance_

# Eigenvectors (component loadings)
eigenvectors = pca.components_
```

## Choosing Number of Components

### Method 1: Explained Variance Ratio
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Fit PCA with all components
pca = PCA()
pca.fit(X_scaled)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.legend()
plt.grid(True)
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

**Common Thresholds:**
- 90% variance: Good for most applications
- 95% variance: Recommended for important features
- 99% variance: Very conservative, minimal information loss

### Method 2: Scree Plot (Elbow Method)
```python
# Plot individual variance per component
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.grid(True)
plt.show()
```

**Look for:** The "elbow" where variance drops significantly

### Method 3: Kaiser Criterion
```python
# Keep components with eigenvalue > 1
eigenvalues = pca.explained_variance_
n_components_kaiser = np.sum(eigenvalues > 1)
print(f"Components with eigenvalue > 1: {n_components_kaiser}")
```

**Rule:** Only keep components explaining more variance than a single original feature

### Method 4: Fixed Number
```python
# Reduce to specific dimension (e.g., for visualization)
pca_2d = PCA(n_components=2)  # For 2D plots
pca_3d = PCA(n_components=3)  # For 3D plots
```

## Complete PCA Workflow

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Standardize features (CRITICAL!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Determine optimal number of components
pca_full = PCA()
pca_full.fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1

print(f"Original dimensions: {X.shape[1]}")
print(f"Components for 95% variance: {n_components}")

# 3. Apply PCA with optimal components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Use reduced data for ML
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_pca, y)
```

## Feature Scaling - CRITICAL!

**PCA is extremely sensitive to feature scale!**

```python
from sklearn.preprocessing import StandardScaler

# ALWAYS scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Then apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Why?** Features with larger variance will dominate the principal components.

**Example:**
- Feature 1: Income (range 20,000-200,000)
- Feature 2: Age (range 18-80)
→ Without scaling, income will dominate PC1

## Interpreting Components

### Component Loadings
```python
import pandas as pd

# Get feature loadings for each component
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)

print(loadings)

# Visualize loadings
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Component Loadings')
plt.tight_layout()
plt.show()
```

**Interpretation:**
- Large positive loading: Feature positively contributes to component
- Large negative loading: Feature negatively contributes to component
- Near zero: Feature doesn't contribute much to component

### Top Contributing Features
```python
# Find top features for PC1
pc1_loadings = abs(pca.components_[0])
top_features_idx = pc1_loadings.argsort()[-5:][::-1]
top_features = [feature_names[i] for i in top_features_idx]

print(f"Top 5 features for PC1: {top_features}")
```

## Visualization

### 2D Visualization
```python
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

# Plot with labels (if available)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                      c=y, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA: First Two Principal Components')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.show()
```

### 3D Visualization
```python
from mpl_toolkits.mplot3d import Axes3D

# Reduce to 3D
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X_scaled)

# 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=y, cmap='viridis', alpha=0.6)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
plt.title('PCA: First Three Principal Components')
plt.colorbar(scatter, label='Class')
plt.show()
```

### Biplot (Features + Samples)
```python
def biplot(X_pca, pca, feature_names):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot samples
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    
    # Plot feature vectors
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, 
                pca.components_[0, i] * 3, 
                pca.components_[1, i] * 3,
                head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.text(pca.components_[0, i] * 3.2, 
               pca.components_[1, i] * 3.2,
               feature, color='red', fontsize=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA Biplot')
    ax.grid(True, alpha=0.3)
    plt.show()

biplot(X_pca_2d, pca, feature_names)
```

## Inverse Transform (Reconstruction)

```python
# Transform to PCA space
X_pca = pca.transform(X_scaled)

# Reconstruct original space
X_reconstructed = pca.inverse_transform(X_pca)

# Calculate reconstruction error
from sklearn.metrics import mean_squared_error
reconstruction_error = mean_squared_error(X_scaled, X_reconstructed)
print(f"Reconstruction error: {reconstruction_error:.4f}")
```

**Use case:** Compression, denoising, anomaly detection

## PCA Variants

### Kernel PCA (Non-linear)
```python
from sklearn.decomposition import KernelPCA

# Apply kernel trick for non-linear relationships
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)
```

**Kernels:**
- `'rbf'`: Radial basis function (most common)
- `'poly'`: Polynomial
- `'sigmoid'`: Sigmoid
- `'cosine'`: Cosine similarity

### Sparse PCA
```python
from sklearn.decomposition import SparsePCA

# Encourage sparse loadings (easier interpretation)
spca = SparsePCA(n_components=2, alpha=1.0)
X_spca = spca.fit_transform(X_scaled)
```

**Benefit:** Components use fewer features (more interpretable)

### Incremental PCA
```python
from sklearn.decomposition import IncrementalPCA

# For datasets too large to fit in memory
ipca = IncrementalPCA(n_components=2, batch_size=100)
X_ipca = ipca.fit_transform(X_scaled)
```

**Use when:** Dataset doesn't fit in RAM

## Common Applications

### 1. Dimensionality Reduction for ML
```python
# Before ML model
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_scaled)

# Train faster model with less overfitting
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_reduced, y)
```

### 2. Data Visualization
```python
# Visualize high-dimensional data in 2D/3D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.show()
```

### 3. Feature Extraction
```python
# Use principal components as new features
pca = PCA(n_components=10)
X_features = pca.fit_transform(X_scaled)

# Combine with original features
X_combined = np.hstack([X, X_features])
```

### 4. Noise Reduction
```python
# Keep major components, discard noise
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_denoised = pca.inverse_transform(X_pca)
```

### 5. Anomaly Detection
```python
# Large reconstruction error = anomaly
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

# Calculate reconstruction error per sample
errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
threshold = np.percentile(errors, 95)
anomalies = errors > threshold
```

### 6. Image Compression
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load grayscale image as matrix
image = plt.imread('image.jpg', format='jpeg')
if len(image.shape) == 3:
    image = image.mean(axis=2)  # Convert to grayscale

# Apply PCA
pca = PCA(n_components=50)  # Keep 50 components
image_pca = pca.fit_transform(image)
image_compressed = pca.inverse_transform(image_pca)

# Display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original')
ax2.imshow(image_compressed, cmap='gray')
ax2.set_title(f'Compressed ({pca.n_components_} components)')
plt.show()
```

## When to Use

✅ **High-dimensional data** (many features)  
✅ **Multicollinearity issues** (correlated features)  
✅ **Speed up training** (reduce features)  
✅ **Visualization** (reduce to 2D/3D)  
✅ **Remove noise** (keep main components)  
✅ **Feature extraction** (create new features)  
✅ **Curse of dimensionality** (too many features vs samples)  

❌ **Non-linear relationships** (use Kernel PCA or t-SNE)  
❌ **Interpretability critical** (loses original feature meaning)  
❌ **Categorical data** (use other methods)  
❌ **Small feature sets** (< 10 features, usually not needed)  
❌ **Tree-based models** (they handle high dimensions well)  

## PCA vs Other Dimensionality Reduction

| Aspect | PCA | t-SNE | UMAP | LDA |
|--------|-----|-------|------|-----|
| **Type** | Linear | Non-linear | Non-linear | Linear |
| **Supervised** | No | No | No | Yes |
| **Speed** | Fast | Slow | Fast | Fast |
| **Scalability** | Excellent | Poor | Good | Good |
| **Preserves** | Global structure | Local structure | Both | Class separation |
| **Use case** | General | Visualization | Visualization | Classification |
| **Deterministic** | Yes | No | No | Yes |

## Best Practices

1. **Always standardize** - Use StandardScaler before PCA
2. **Check explained variance** - Ensure you keep enough information
3. **Visualize scree plot** - Understand variance distribution
4. **Interpret components** - Look at loadings for insights
5. **Validate on test set** - Fit on train, transform on test
6. **Don't use for tree models** - Trees handle high dimensions well
7. **Consider alternatives** - Try t-SNE/UMAP for visualization
8. **Document components kept** - For reproducibility

## Common Pitfalls

❌ **Not scaling features** - Dominates by large-scale features  
❌ **Applying to test set incorrectly** - Must use .transform(), not .fit_transform()  
❌ **Ignoring explained variance** - May lose important information  
❌ **Using with categorical data** - Not appropriate  
❌ **Over-reducing dimensions** - Can hurt model performance  
❌ **Interpreting components as features** - They're combinations of features  

## Complete Example: ML Pipeline with PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Create pipeline (ensures proper train/test handling)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('classifier', LogisticRegression())
])

# 3. Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 4. Train final model
pipeline.fit(X_train, y_train)

# 5. Evaluate
test_score = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_score:.3f}")

# 6. Inspect PCA
pca = pipeline.named_steps['pca']
print(f"Original features: {X_train.shape[1]}")
print(f"PCA components: {pca.n_components_}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

## Advanced: Whitening

```python
# PCA with whitening (decorrelate and scale to unit variance)
pca = PCA(n_components=10, whiten=True)
X_whitened = pca.fit_transform(X_scaled)

# Check: components have unit variance
print(f"Component variances: {X_whitened.var(axis=0)}")
```

**Benefit:** All components have equal importance (variance = 1)

---

*PCA is the most fundamental dimensionality reduction technique. Master it to handle high-dimensional data, speed up training, and create powerful visualizations!*
