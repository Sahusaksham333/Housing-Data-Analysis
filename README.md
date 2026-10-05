from pathlib import Path

readme = r"""# California Housing Price Prediction

## 1. Project Overview

**California Housing Price Prediction** is a supervised machine learning project focused on predicting the **median house value of districts in California** using demographic, geographic, housing, and income-related attributes.

The project implements and evaluates multiple regression algorithms and examines the relationship between **median household income** and **median house value**.

The project is based on the California Census housing dataset and follows a structured machine learning workflow covering data preprocessing, feature transformation, model development, evaluation, and visualization.

---

## 2. Problem Statement

The objective is to develop a machine learning model capable of predicting the median housing price of a California district from the available district-level attributes.

Given information such as:

- Geographic coordinates
- Housing age
- Number of rooms and bedrooms
- Population
- Number of households
- Median income
- Ocean proximity

the model learns the relationship between these characteristics and the **median house value**.

### Project Objectives

1. Build a regression model to predict median house values.
2. Prepare the dataset through appropriate preprocessing techniques.
3. Train and evaluate multiple regression algorithms.
4. Compare model performance using **Root Mean Squared Error (RMSE)**.
5. Analyze the relationship between `median_income` and `median_house_value`.
6. Visualize the fitted linear regression model for the single-feature analysis.

---

## 3. Dataset

The dataset consists of **20,640 observations and 10 variables** representing California districts.

### Dataset Features

| Feature | Description |
|---|---|
| `longitude` | Longitude of the district/block in California |
| `latitude` | Latitude of the district/block in California |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms, excluding bedrooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Total population of the district |
| `households` | Total number of households |
| `median_income` | Median household income |
| `ocean_proximity` | Categorical indicator representing the district's proximity to the ocean |
| `median_house_value` | Median household value and prediction target |

### `ocean_proximity` Categories

The categorical variable contains the following values:

- `NEAR BAY`
- `<1H OCEAN`
- `INLAND`
- `NEAR OCEAN`
- `ISLAND`

---

## 4. Machine Learning Workflow

The project follows the workflow below:

```text
                    California Housing Dataset
                              │
                              ▼
                     Data Loading & Inspection
                              │
                              ▼
                    Feature / Target Separation
                              │
                              ▼
                      Missing Value Handling
                              │
                              ▼
                     Categorical Encoding
                              │
                              ▼
                       Train-Test Split
                         80% / 20%
                              │
                              ▼
                      Feature Standardization
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
          Linear Regression  Decision Tree  Random Forest
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                       RMSE Evaluation
                              │
                              ▼
                  Median Income Regression
                              │
                              ▼
                       Data Visualization
```

---

## 5. Data Preprocessing

### 5.1 Missing Value Handling

Missing numerical values are replaced using the mean of the respective numerical columns.

```python
X = X.fillna(X.mean(numeric_only=True))
```

This ensures that missing numerical observations do not prevent model training.

### 5.2 Categorical Encoding

The `ocean_proximity` feature is categorical and is converted into numerical representation before model training.

The current implementation uses `LabelEncoder`.

### 5.3 Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**
- `random_state = 50`

The training set is used to fit the models, while the test set is reserved for evaluating their predictive performance.

### 5.4 Feature Standardization

`StandardScaler` is used to standardize the feature variables.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardization places numerical features on a comparable scale and is particularly relevant for the Linear Regression workflow.

---

## 6. Regression Models

### 6.1 Linear Regression

Linear Regression is implemented as the baseline regression model.

The model learns a linear relationship between the input features and `median_house_value`.

**Reported RMSE:**

```text
69,321.01
```

### 6.2 Decision Tree Regression

A `DecisionTreeRegressor` is trained using the processed training dataset and subsequently evaluated on the test dataset.

**Reported RMSE:**

```text
69,416.88
```

### 6.3 Random Forest Regression

A `RandomForestRegressor` with 100 estimators is included as an additional regression approach.

The current notebook contains the Random Forest model definition; however, its final evaluation loop requires correction before a Random Forest-specific RMSE can be reported reliably.

Accordingly, no Random Forest performance value is presented as a validated result in this README.

---

## 7. Model Evaluation

The primary evaluation metric used in the project is **Root Mean Squared Error (RMSE)**.

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

where:

- \(y_i\) represents the actual value.
- \(\hat{y}_i\) represents the predicted value.
- \(n\) represents the number of observations.

A lower RMSE indicates a smaller average magnitude of prediction error.

### Current Results

| Model | RMSE |
|---|---:|
| Linear Regression | 69,321.01 |
| Decision Tree Regression | 69,416.88 |
| Random Forest Regression | Not validated in the current implementation |

The current results show that the Linear Regression and Decision Tree models have comparable test-set RMSE values.

---

## 8. Median Income Analysis

As an additional analysis, the project performs Linear Regression using only:

```text
median_income
```

as the independent variable.

The objective is to examine the relationship between household income and median house value.

The analysis includes:

1. Extracting `median_income` from the training and testing feature sets.
2. Training a Linear Regression model using this single feature.
3. Predicting median house values.
4. Plotting the fitted regression relationship for the training and testing observations.

This provides a simplified view of how strongly household income is associated with housing values without incorporating the remaining dataset features.

---

## 9. Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, manipulation, and preprocessing |
| **NumPy** | Numerical computation |
| **Matplotlib** | Data visualization |
| **Scikit-learn** | Machine learning, preprocessing, and evaluation |
| **Jupyter Notebook** | Interactive development and analysis |

---

## 10. Project Structure

```text
California-Housing-Price-Prediction/
│
├── Housing-Data-Analysis.ipynb
├── housing.csv
├── Housing-Problem.pdf
└── README.md
```

---

## 11. Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd California-Housing-Price-Prediction
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

---

## 12. Execution

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Housing-Data-Analysis.ipynb
```

Ensure that `housing.csv` is available in the appropriate working directory.

Execute the notebook cells sequentially to reproduce the analysis and model evaluation.

---

## 13. Key Learning Outcomes

This project demonstrates practical implementation of:

- Supervised machine learning
- Regression analysis
- Data preprocessing
- Missing-value treatment
- Categorical feature encoding
- Train-test data partitioning
- Feature standardization
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- RMSE-based model evaluation
- Single-variable regression analysis
- Regression visualization

---

## 14. Future Improvements

The project can be further enhanced through the following improvements:

1. Correctly integrate Random Forest into the final model evaluation pipeline.
2. Use `OneHotEncoder` for the nominal `ocean_proximity` feature.
3. Implement `Pipeline` and `ColumnTransformer` for reproducible preprocessing.
4. Evaluate additional metrics such as **MAE** and **R² Score**.
5. Apply cross-validation for more robust model assessment.
6. Perform hyperparameter optimization using `GridSearchCV` or `RandomizedSearchCV`.
7. Analyze feature importance using tree-based models.
8. Expand the exploratory data analysis with correlation and distribution analysis.
9. Persist the selected model for future predictions.
10. Develop an inference interface for predicting housing values from new observations.

---

## 15. Conclusion

This project demonstrates an end-to-end introductory machine learning workflow for **California housing price prediction**.

The analysis begins with data preparation and preprocessing, followed by the development of regression models and evaluation using RMSE. The additional single-variable analysis of `median_income` provides a focused examination of its relationship with `median_house_value`.

The project establishes a foundation for more advanced housing-price prediction systems involving robust preprocessing pipelines, ensemble methods, hyperparameter optimization, cross-validation, feature engineering, and model deployment.

---

## Project Information

**Project:** California Housing Price Prediction  
**Domain:** Finance & Housing  
**Problem Type:** Supervised Regression  
**Primary Target:** `median_house_value`  
**Dataset Size:** 20,640 × 10  
**Primary Evaluation Metric:** RMSE
"""

out = Path("/mnt/data/README.md")
out.write_text(readme, encoding="utf-8")
print(f"Created: {out}")
