# Linear Regression

A fundamental algorithm that models the relationship between input features and a continuous output variable using a linear equation.

## How It Works

Linear Regression finds the best-fitting straight line (or hyperplane) through the data:

1. **Define the model** - y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
2. **Minimize error** - Find coefficients (β) that minimize prediction errors
3. **Make predictions** - Use learned coefficients to predict new values

**Goal:** Minimize the sum of squared residuals (difference between actual and predicted values)

## Key Characteristics

**Strengths:**
- Simple, fast, and interpretable
- Low computational requirements
- Works well with linearly separable data
- Provides coefficient significance tests
- Extrapolates beyond training data

**Weaknesses:**
- Assumes linear relationship (limited flexibility)
- Sensitive to outliers
- Assumes independence of errors
- Multicollinearity can be problematic
- Cannot handle non-linear patterns without feature engineering

## Quick Example

```python
from sklearn.linear_model import LinearRegression

# Create and train
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
predictions = lr.predict(X_test)

# Model parameters
print(f"Intercept: {lr.intercept_}")
print(f"Coefficients: {lr.coef_}")
```

## Model Equation

**Simple Linear Regression (1 feature):**
```
y = β₀ + β₁x
```

**Multiple Linear Regression (n features):**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- `y` = predicted value (dependent variable)
- `x` = features (independent variables)
- `β₀` = intercept (bias term)
- `β₁, β₂, ...` = coefficients (weights)

## Cost Function

**Mean Squared Error (MSE):**
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Optimization Method:**
- **Normal Equation** (closed-form solution) - Fast for small datasets
- **Gradient Descent** - Better for large datasets

## Assumptions

Linear Regression assumes:

1. **Linearity** - Relationship between X and y is linear
2. **Independence** - Observations are independent
3. **Homoscedasticity** - Constant variance of residuals
4. **Normality** - Residuals are normally distributed
5. **No multicollinearity** - Features are not highly correlated

## Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"R² Score: {r2:.3f}")  # Proportion of variance explained
print(f"RMSE: {rmse:.3f}")    # Root Mean Squared Error
print(f"MAE: {mae:.3f}")      # Mean Absolute Error
```

**R² Score (Coefficient of Determination):**
- Range: 0 to 1 (can be negative for poor models)
- 1.0 = Perfect predictions
- 0.0 = Model no better than predicting the mean

## Regularization Variants

### Ridge Regression (L2)
```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha controls regularization strength
ridge.fit(X_train, y_train)
```
- Adds penalty: `α Σβ²`
- Shrinks coefficients toward zero
- Good when features are correlated

### Lasso Regression (L1)
```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
```
- Adds penalty: `α Σ|β|`
- Can set coefficients to exactly zero (feature selection)
- Good for sparse models

### Elastic Net (L1 + L2)
```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```
- Combines Ridge and Lasso
- `l1_ratio` controls L1 vs L2 balance

## Feature Engineering

Linear Regression can handle non-linear relationships through transformations:

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Now Linear Regression can fit curves!
lr.fit(X_poly, y)
```

**Common transformations:**
- Polynomial features (x², x³)
- Interaction terms (x₁ × x₂)
- Log transforms (log(x))
- Square root transforms (√x)

## When to Use

✅ Understanding feature relationships  
✅ Fast predictions needed  
✅ Linear relationships in data  
✅ Interpretability is critical  
✅ Baseline model before trying complex algorithms  
❌ Complex non-linear patterns  
❌ Heavy outliers in data  
❌ High-dimensional data with correlated features  

## Best Practices

1. **Check assumptions** - Plot residuals to verify assumptions hold
2. **Scale features** - Use StandardScaler for better convergence
3. **Handle outliers** - Remove or use robust regression methods
4. **Feature selection** - Remove correlated features or use regularization
5. **Cross-validation** - Always validate on held-out data
6. **Start simple** - Use Linear Regression as a baseline before complex models

## Diagnostic Plots

```python
import matplotlib.pyplot as plt

# Residual plot
residuals = y_test - predictions
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

**Good model:** Residuals randomly scattered around zero  
**Bad model:** Pattern in residuals (curved, funnel-shaped)

---

*Linear Regression is the foundation of predictive modeling. Master it first, then move to more complex algorithms.*
