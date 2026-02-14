# Decision Trees & Random Forest

A comprehensive guide to tree-based machine learning algorithms: from single trees to powerful ensembles.

---

## Decision Trees

A tree-like model that makes decisions by asking a series of questions about the data.

### How It Works

Decision Trees split data into branches based on feature values, creating a flowchart-like structure:

1. **Start at root** - Select the best feature to split on
2. **Create branches** - Split data based on feature threshold
3. **Repeat recursively** - Continue splitting until stopping criteria met
4. **Leaf nodes** - Final predictions (class label or numeric value)

### Key Characteristics

**Strengths:**
- Easy to visualize and interpret (white-box model)
- Handles both numerical and categorical data
- No feature scaling needed
- Captures non-linear relationships

**Weaknesses:**
- Prone to overfitting (high variance)
- Unstable - small data changes create different trees
- Biased toward dominant classes
- Can create overly complex trees

### Quick Example

```python
from sklearn.tree import DecisionTreeClassifier

# Create and train
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
dt.fit(X_train, y_train)

# Predict
predictions = dt.predict(X_test)

# Visualize
from sklearn.tree import plot_tree
plot_tree(dt, feature_names=feature_names, filled=True)
```

### Splitting Criteria

**Classification:**
- **Gini Impurity**: Measures probability of incorrect classification (default)
- **Entropy**: Information gain based on Shannon entropy

**Regression:**
- **MSE**: Mean Squared Error minimization
- **MAE**: Mean Absolute Error (more robust to outliers)

### Controlling Overfitting

```python
DecisionTreeClassifier(
    max_depth=10,              # Limit tree depth
    min_samples_split=20,      # Min samples to split node
    min_samples_leaf=10,       # Min samples in leaf
    max_leaf_nodes=50,         # Limit total leaves
    min_impurity_decrease=0.01 # Min improvement to split
)
```

---

## Random Forest

An ensemble learning method that combines multiple decision trees to create a more robust and accurate model.

### How It Works

Random Forest builds many decision trees and aggregates their predictions:

1. **Bootstrap sampling** - Create multiple random subsets of training data (with replacement)
2. **Random feature selection** - Each split considers only a random subset of features
3. **Build trees** - Train independent decision trees on each bootstrap sample
4. **Aggregate predictions** - Classification: majority vote | Regression: average

### Key Characteristics

**Strengths:**
- High accuracy with low overfitting risk
- Handles thousands of features without feature selection
- Estimates feature importance
- Robust to outliers and noise
- Minimal hyperparameter tuning needed

**Weaknesses:**
- Less interpretable than single decision tree
- Slower training and prediction than single tree
- Larger memory footprint
- Can be biased toward dominant classes

### Quick Example

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Tree depth
    min_samples_split=5,   # Min samples to split
    random_state=42
)
rf.fit(X_train, y_train)

# Predict
predictions = rf.predict(X_test)

# Feature importance
importances = rf.feature_importances_
```

### Key Hyperparameters

```python
RandomForestClassifier(
    n_estimators=100,          # More trees = better (diminishing returns after ~100-500)
    max_features='sqrt',       # Features per split (sqrt for classification, 1/3 for regression)
    max_depth=None,            # Tree depth (None = grow until pure)
    min_samples_split=2,       # Min samples to split node
    min_samples_leaf=1,        # Min samples in leaf
    bootstrap=True,            # Use bootstrap sampling
    oob_score=True,            # Out-of-bag score for validation
    n_jobs=-1                  # Parallel processing
)
```

### Why Random Forest Works So Well

**Bias-Variance Trade-off:**
- Individual trees: Low bias, high variance (prone to overfitting)
- Random Forest: Low bias, reduced variance (averaging reduces overfitting)

**Two sources of randomness:**
1. **Bagging** - Different training subsets
2. **Feature randomness** - Different feature subsets per split

Result: Diverse trees that make different errors → averaging cancels out errors

### Feature Importance

```python
# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

Measures how much each feature decreases impurity across all trees.

### Out-of-Bag (OOB) Score

Free validation without separate test set:

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_}")
```

Each tree is tested on samples it didn't see during training (~37% of data).

---

## Decision Tree vs Random Forest

| Aspect | Decision Tree | Random Forest |
|--------|--------------|---------------|
| **Accuracy** | Good | Excellent |
| **Overfitting** | High risk | Low risk |
| **Interpretability** | Very high | Low to medium |
| **Training time** | Fast | Slower |
| **Prediction time** | Very fast | Moderate |
| **Stability** | Unstable | Very stable |
| **Feature importance** | Available | More reliable |
| **Memory usage** | Low | High |

## When to Use Each

### Use Decision Tree When:
✅ Interpretability is critical  
✅ Need to explain predictions to stakeholders  
✅ Fast predictions required  
✅ Exploratory analysis phase  
✅ Small to medium datasets  

### Use Random Forest When:
✅ **Default choice** for tabular data  
✅ Accuracy is the priority  
✅ High-dimensional datasets  
✅ Need robust feature importance  
✅ Production machine learning systems  

### Avoid Both When:
❌ Deep learning suited (images, text, sequences)  
❌ Extremely large datasets (use XGBoost/LightGBM)  
❌ Real-time predictions with strict latency requirements  

---

## Best Practices

### Decision Trees
- **Always set constraints** to prevent overfitting
- **Use cross-validation** to tune hyperparameters
- **Prune trees** after training for better generalization
- **Consider ensembles** for production use

### Random Forest
- Start with **100 trees**, increase if needed
- Use **n_jobs=-1** for parallel processing
- Set **max_features='sqrt'** for classification
- Use **oob_score=True** to monitor performance
- Limit **max_depth** for very large datasets

### General Tips
- Decision Trees are great for **understanding** your data
- Random Forest is great for **predicting** on your data
- Both handle mixed data types (no preprocessing needed)
- Neither requires feature scaling
- For further improvements, consider **XGBoost** or **LightGBM**

---

*Decision Trees are the building blocks. Random Forest is the production-ready powerhouse. Master both to understand modern ensemble methods.*
