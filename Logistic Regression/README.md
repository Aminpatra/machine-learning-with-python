# Logistic Regression

A classification algorithm that predicts the probability of a binary outcome using a logistic (sigmoid) function.

## How It Works

Despite its name, Logistic Regression is used for **classification**, not regression:

1. **Linear combination** - Calculate z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
2. **Apply sigmoid** - Transform z to probability: p = 1 / (1 + e⁻ᶻ)
3. **Make prediction** - If p ≥ 0.5 → Class 1, else → Class 0
4. **Optimize** - Find coefficients that maximize likelihood

## Key Characteristics

**Strengths:**
- Fast training and prediction
- Outputs calibrated probabilities
- Interpretable coefficients (log-odds)
- Works well with linearly separable classes
- No hyperparameters to tune (basic version)
- Handles multi-class (one-vs-rest or multinomial)

**Weaknesses:**
- Assumes linear decision boundary
- Sensitive to outliers
- Requires feature engineering for non-linear patterns
- Struggles with highly correlated features
- Poor performance on imbalanced data (without adjustments)

## Quick Example

```python
from sklearn.linear_model import LogisticRegression

# Create and train
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict classes
predictions = log_reg.predict(X_test)

# Predict probabilities
probabilities = log_reg.predict_proba(X_test)

# Model parameters
print(f"Intercept: {log_reg.intercept_}")
print(f"Coefficients: {log_reg.coef_}")
```

## The Sigmoid Function

**Formula:**
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Properties:**
- Maps any value to range (0, 1)
- S-shaped curve
- Output interpreted as probability
- z = 0 → p = 0.5 (decision boundary)

**Model Equation:**
```
P(y=1|x) = 1 / (1 + e⁻⁽β₀ + β₁x₁ + ... + βₙxₙ⁾)
```

## Cost Function

**Log Loss (Binary Cross-Entropy):**
```
Cost = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Optimization:**
- Gradient Descent (most common)
- Newton's Method (faster, more memory)
- No closed-form solution like Linear Regression

## Interpreting Coefficients

Coefficients represent **log-odds** (logit):

```python
# Positive coefficient → increases probability of class 1
# Negative coefficient → decreases probability of class 1
# Coefficient magnitude → strength of effect

# Convert to odds ratio
import numpy as np
odds_ratio = np.exp(log_reg.coef_)
```

**Example:**
- Coefficient = 0.5 → Odds ratio = e^0.5 ≈ 1.65
- One unit increase in feature multiplies odds by 1.65

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Basic metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Probability-based metric
auc = roc_auc_score(y_test, probabilities[:, 1])

# Detailed report
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
```

**Key Metrics:**
- **Accuracy** - Overall correctness (use when classes balanced)
- **Precision** - Of predicted positives, how many are correct
- **Recall** - Of actual positives, how many found
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve (threshold-independent)

## Multi-Class Classification

```python
# One-vs-Rest (default)
log_reg = LogisticRegression(multi_class='ovr')

# Multinomial (softmax)
log_reg = LogisticRegression(multi_class='multinomial')
```

**One-vs-Rest:** Train separate binary classifier for each class  
**Multinomial:** Single model with softmax function (better for related classes)

## Regularization

Control overfitting and handle correlated features:

```python
# L2 Regularization (Ridge - default)
log_reg = LogisticRegression(penalty='l2', C=1.0)

# L1 Regularization (Lasso - feature selection)
log_reg = LogisticRegression(penalty='l1', solver='saga', C=1.0)

# Elastic Net (L1 + L2)
log_reg = LogisticRegression(penalty='elasticnet', solver='saga', 
                              C=1.0, l1_ratio=0.5)

# No regularization
log_reg = LogisticRegression(penalty=None)
```

**C Parameter:**
- Inverse of regularization strength
- Smaller C → stronger regularization → simpler model
- Larger C → weaker regularization → more complex model
- Default: C = 1.0

## Handling Imbalanced Data

```python
# Class weights (penalize misclassification of minority class more)
log_reg = LogisticRegression(class_weight='balanced')

# Manual weights
log_reg = LogisticRegression(class_weight={0: 1, 1: 10})

# Adjust decision threshold
threshold = 0.3  # Lower for imbalanced data
predictions = (probabilities[:, 1] >= threshold).astype(int)
```

## Feature Scaling

**Important:** Always scale features for Logistic Regression!

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg.fit(X_train_scaled, y_train)
```

Scaling ensures:
- Faster convergence
- Coefficients are comparable
- Better regularization performance

## Advanced Options

```python
LogisticRegression(
    penalty='l2',              # Regularization type
    C=1.0,                     # Inverse regularization strength
    solver='lbfgs',            # Optimization algorithm
    max_iter=100,              # Maximum iterations
    multi_class='auto',        # Multi-class strategy
    class_weight='balanced',   # Handle imbalanced data
    random_state=42,           # Reproducibility
    n_jobs=-1                  # Parallel processing
)
```

**Solvers:**
- `lbfgs` - Good default, handles L2
- `saga` - Handles L1, Elastic Net, large datasets
- `liblinear` - Good for small datasets, binary problems
- `newton-cg` - Fast for small datasets

## When to Use

✅ Binary or multi-class classification  
✅ Need probability estimates  
✅ Interpretability is important  
✅ Fast training/prediction needed  
✅ Linear decision boundary acceptable  
✅ Baseline model before complex algorithms  
❌ Complex non-linear patterns (use tree-based or neural networks)  
❌ Very high-dimensional sparse data (try Naive Bayes)  
❌ Need to capture feature interactions (add them manually or use non-linear models)  

## Best Practices

1. **Scale features** - Always use StandardScaler or MinMaxScaler
2. **Check for linearity** - Plot feature relationships with target
3. **Handle imbalanced data** - Use class_weight='balanced'
4. **Cross-validate** - Tune C parameter using GridSearchCV
5. **Use probabilities** - Don't just rely on hard predictions
6. **Check multicollinearity** - Remove or regularize correlated features
7. **Start simple** - Use as baseline before trying complex models
8. **Threshold tuning** - Adjust decision threshold based on business needs

## ROC Curve & Threshold Selection

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Get probabilities
probs = log_reg.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Plot
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
```

## Logistic vs Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Task** | Regression (continuous) | Classification (binary/multi-class) |
| **Output** | Real number | Probability (0 to 1) |
| **Function** | Linear | Sigmoid (logistic) |
| **Cost** | MSE | Log Loss |
| **Decision boundary** | N/A | Linear (in feature space) |
| **Assumptions** | Normality, homoscedasticity | None (non-parametric) |

---

*Logistic Regression is the go-to algorithm for classification problems where you need speed, interpretability, and probability estimates.*
