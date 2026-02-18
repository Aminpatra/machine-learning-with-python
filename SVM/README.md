# Support Vector Machine (SVM)

A powerful supervised learning algorithm that finds the optimal hyperplane to separate classes with maximum margin.

## How It Works

SVM creates a decision boundary that maximizes the distance (margin) between classes:

1. **Find support vectors** - Data points closest to the decision boundary
2. **Maximize margin** - Create the widest possible "street" between classes
3. **Optimal hyperplane** - The boundary that maximizes this margin
4. **Kernel trick** - Transform data to higher dimensions for non-linear separation

**Key Insight:** Only support vectors (boundary points) matter; other points can be ignored!

## Key Characteristics

**Strengths:**
- Effective in high-dimensional spaces
- Works well with clear margin of separation
- Memory efficient (only uses support vectors)
- Versatile with different kernel functions
- Robust to overfitting (especially in high dimensions)
- Excellent for binary classification

**Weaknesses:**
- Computationally expensive for large datasets (O(n²) to O(n³))
- Sensitive to feature scaling
- Difficult to interpret (black-box model)
- Choosing the right kernel and hyperparameters can be tricky
- Poor performance on overlapping classes
- Doesn't provide probability estimates directly

## Quick Example

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# CRITICAL: Always scale features for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Predict
predictions = svm.predict(X_test_scaled)

# Number of support vectors
print(f"Support vectors: {svm.n_support_}")
```

## The Margin Concept

**Hard Margin (Linearly Separable Data):**
- Perfect separation with no misclassifications
- Only works when data is perfectly separable
- Sensitive to outliers

**Soft Margin (Real-World Data):**
- Allows some misclassifications (controlled by C parameter)
- More robust to outliers
- Balances margin width vs. classification errors

**Decision Function:**
```
f(x) = sign(w·x + b)
```
- `w` = weight vector (perpendicular to hyperplane)
- `b` = bias term
- Points on hyperplane: w·x + b = 0

## Kernel Functions

Kernels allow SVM to handle non-linear decision boundaries by mapping data to higher dimensions.

### Linear Kernel
```python
svm = SVC(kernel='linear')
```
- **Use when:** Data is linearly separable
- **Equation:** K(x, x') = x · x'
- **Pros:** Fast, interpretable, works well in high dimensions
- **Cons:** Cannot handle non-linear patterns

### RBF (Radial Basis Function) Kernel - Most Popular
```python
svm = SVC(kernel='rbf', gamma='scale')
```
- **Use when:** Non-linear patterns, general-purpose (default choice)
- **Equation:** K(x, x') = exp(-γ||x - x'||²)
- **Pros:** Can handle any shape, most flexible
- **Cons:** Requires tuning gamma, slower

### Polynomial Kernel
```python
svm = SVC(kernel='poly', degree=3)
```
- **Use when:** Polynomial relationships in data
- **Equation:** K(x, x') = (γx · x' + r)^d
- **Pros:** Good for image processing, specific non-linear patterns
- **Cons:** More parameters to tune, can overfit

### Sigmoid Kernel
```python
svm = SVC(kernel='sigmoid')
```
- **Use when:** Neural network-like behavior needed
- **Equation:** K(x, x') = tanh(γx · x' + r)
- **Pros:** Similar to neural networks
- **Cons:** Not positive semi-definite, rarely used

## Key Hyperparameters

### C Parameter (Regularization)
```python
svm = SVC(C=1.0)  # Default
```
- **Controls:** Trade-off between margin width and misclassification
- **Small C (e.g., 0.1):** Wide margin, more misclassifications (underfitting)
- **Large C (e.g., 100):** Narrow margin, fewer misclassifications (overfitting)
- **Tuning range:** Try [0.001, 0.01, 0.1, 1, 10, 100, 1000]

### Gamma Parameter (RBF/Poly kernels)
```python
svm = SVC(gamma='scale')  # Default: 1 / (n_features * X.var())
svm = SVC(gamma='auto')   # Alternative: 1 / n_features
svm = SVC(gamma=0.1)      # Manual value
```
- **Controls:** Influence of single training example
- **Small gamma (e.g., 0.001):** Far reach, smooth boundary (underfitting)
- **Large gamma (e.g., 10):** Close reach, wiggly boundary (overfitting)
- **Tuning range:** Try [0.0001, 0.001, 0.01, 0.1, 1, 10]

### Degree (Polynomial kernel only)
```python
svm = SVC(kernel='poly', degree=3)
```
- Polynomial degree (2, 3, 4, ...)
- Higher degree = more complex boundaries

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'linear']
}

# Grid search with cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Use best model
best_svm = grid_search.best_estimator_
```

## Multi-Class Classification

SVM is inherently binary, but handles multi-class automatically:

**One-vs-One (default):**
```python
svm = SVC(decision_function_shape='ovo')
```
- Trains n(n-1)/2 classifiers (one for each pair)
- Prediction: majority vote
- More training time, but often better accuracy

**One-vs-Rest:**
```python
svm = SVC(decision_function_shape='ovr')
```
- Trains n classifiers (one per class vs. all others)
- Faster training
- Good for many classes

## Probability Estimates

```python
# Enable probability estimates (slower)
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)

# Get probabilities
probabilities = svm.predict_proba(X_test_scaled)
```

**Note:** Uses Platt scaling (5-fold CV internally), adds training time

## Feature Scaling - CRITICAL!

**SVM is extremely sensitive to feature scale!**

```python
from sklearn.preprocessing import StandardScaler

# Always scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Then train SVM
svm.fit(X_train_scaled, y_train)
```

**Why?** Distance-based algorithm; large-scale features dominate the margin calculation.

## Evaluation Example

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
predictions = svm.predict(X_test_scaled)

# Confusion Matrix
print(confusion_matrix(y_test, predictions))

# Detailed Report
print(classification_report(y_test, predictions))

# Perfect results (like your Iris example):
# [[13  0  0]
#  [ 0 11  0]
#  [ 0  0 21]]
#
# Precision, Recall, F1-Score all = 1.00 for all classes!
```

## When to Use

✅ **Small to medium datasets** (< 10,000 samples)  
✅ **High-dimensional data** (text classification, gene expression)  
✅ **Clear margin between classes**  
✅ **Binary classification problems**  
✅ **Non-linear decision boundaries** (with RBF kernel)  
✅ **When accuracy is critical**  

❌ **Very large datasets** (slow training, use SGDClassifier instead)  
❌ **Noisy data with overlapping classes**  
❌ **When interpretability is critical** (use Decision Trees)  
❌ **Need probability estimates** (slower with SVM)  
❌ **Real-time predictions on large scale** (use simpler models)  

## SVM vs Other Algorithms

| Aspect | SVM | Logistic Regression | Random Forest |
|--------|-----|-------------------|---------------|
| **Speed** | Slow (O(n²)) | Fast | Medium |
| **Non-linear** | Yes (kernels) | No (unless engineered) | Yes |
| **Interpretability** | Low | High | Medium |
| **Scaling needed** | **Critical** | Important | No |
| **High dimensions** | Excellent | Good | Can struggle |
| **Large datasets** | Poor | Excellent | Good |
| **Overfitting** | Low (with tuning) | Low | Very low |

## Best Practices

1. **Always scale features** - Use StandardScaler or MinMaxScaler
2. **Start with RBF kernel** - Best general-purpose choice
3. **Use GridSearchCV** - Tune C and gamma together
4. **Cross-validate** - SVM sensitive to train/test split
5. **Check support vectors** - Too many? Try smaller C or different kernel
6. **Handle imbalanced data** - Use `class_weight='balanced'`
7. **Large datasets** - Use `LinearSVC` or `SGDClassifier` instead
8. **Feature selection** - Remove irrelevant features for better performance

## Advanced: Support Vector Regression (SVR)

SVM can also do regression:

```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
predictions = svr.predict(X_test_scaled)
```

## Performance Optimization

### For Large Datasets:
```python
from sklearn.svm import LinearSVC  # Much faster for linear kernel
from sklearn.linear_model import SGDClassifier  # Online learning

# LinearSVC (faster linear SVM)
linear_svm = LinearSVC(C=1.0, max_iter=1000)

# SGD with hinge loss (approximates linear SVM)
sgd = SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000)
```

### Memory-Efficient:
```python
# Limit cache size
svm = SVC(cache_size=200)  # MB of cache

# Use LinearSVC for linear problems
linear_svm = LinearSVC()
```

## Common Pitfalls

❌ **Forgetting to scale** - Results will be poor  
❌ **Using default parameters** - Always tune C and gamma  
❌ **Trying SVM on huge datasets** - Use alternatives  
❌ **Ignoring kernel choice** - Wrong kernel = poor performance  
❌ **Not handling imbalanced classes** - Use class_weight  

## Quick Start Recipe

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 1. Scale features (CRITICAL!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# 3. Grid search
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

# 4. Evaluate
best_svm = grid.best_estimator_
accuracy = best_svm.score(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

---

*SVM is a powerful algorithm that can achieve perfect classification (like your Iris results!) when properly tuned. Master feature scaling and hyperparameter tuning for best results.*
