# K-Nearest Neighbors (KNN)

A simple, intuitive machine learning algorithm for classification and regression based on proximity.

## How It Works

KNN makes predictions by finding the **k closest data points** to a new sample and using their values to determine the output:

1. **Calculate distances** between the new point and all training samples
2. **Select k nearest neighbors** based on distance metric (typically Euclidean)
3. **Vote or average** - Classification: majority vote | Regression: mean value

## Key Characteristics

**Strengths:**
- No training phase (lazy learning)
- Naturally handles multi-class problems
- Simple to understand and implement

**Weaknesses:**
- Slow prediction time on large datasets
- Sensitive to irrelevant features and scale
- Requires choosing k and distance metric

## Quick Example

```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)
```

## Choosing K

- **Small k** (e.g., 1-3): Sensitive to noise, complex decision boundaries
- **Large k** (e.g., 10-20): Smoother boundaries, may miss local patterns
- **Best practice**: Use cross-validation; often k = √n is a good start

## When to Use

✅ Small to medium datasets  
✅ Low-dimensional feature spaces  
✅ Non-linear decision boundaries  
❌ High-dimensional data (curse of dimensionality)  
❌ Real-time predictions needed  

## Essential Preprocessing

**Feature scaling is critical** - use StandardScaler or MinMaxScaler to ensure all features contribute equally to distance calculations.

---

*KNN is perfect for getting started with ML, but consider tree-based methods or neural networks for production systems with large datasets.*
