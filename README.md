from pathlib import Path

readme = r"""# California Housing Price Prediction

A machine learning regression project that predicts **median house values in California** using demographic, geographic, housing, and income-related features from the California Census housing dataset.

## Project Overview

The objective of this project is to build regression models capable of predicting the median housing value of a California district based on the remaining available metrics.

The project follows a complete introductory machine-learning workflow:

- Load and inspect the housing dataset
- Separate features and target variable
- Handle missing numerical values
- Encode categorical data
- Split data into training and testing sets
- Standardize numerical features
- Train multiple regression models
- Evaluate models using **Root Mean Squared Error (RMSE)**
- Analyze the relationship between **median income** and **median house value**
- Visualize a one-variable linear regression model

The project specification identifies the domain as **Finance and Housing** and describes 20,640 districts with 10 variables.

## Dataset

The dataset contains **20,640 rows and 10 columns**.

### Features

| Feature | Description |
|---|---|
| `longitude` | Longitude of the block in California |
| `latitude` | Latitude of the block in California |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total number of rooms, excluding bedrooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Total population in the block |
| `households` | Total number of households |
| `median_income` | Median household income |
| `ocean_proximity` | Categorical indicator describing proximity to the ocean |
| `median_house_value` | Median household value; prediction target |

### Categorical Values

`ocean_proximity` contains the following categories:

- `NEAR BAY`
- `<1H OCEAN`
- `INLAND`
- `NEAR OCEAN`
- `ISLAND`

## Machine Learning Workflow

```text
Raw Housing Data
       │
       ▼
Load Dataset
       │
       ▼
Feature / Target Separation
       │
       ▼
Missing-Value Handling
       │
       ▼
Categorical Encoding
       │
       ▼
80/20 Train-Test Split
       │
       ▼
Feature Standardization
       │
       ├───────────────┐
       ▼               ▼
Linear Regression   Decision Tree
       │               │
       └───────┬───────┘
               ▼
        RMSE Evaluation
               │
               ▼
       Median-Income Analysis
```

## Models

### 1. Linear Regression

A baseline regression model is trained using all processed features.

The notebook reports:

```text
Linear Regression RMSE: 69321.01
```

### 2. Decision Tree Regression

A `DecisionTreeRegressor` is trained on the standardized training data and evaluated on the test set.

The notebook reports an RMSE of approximately:

```text
Decision Tree RMSE: 69416.88
```

### 3. Random Forest Regression

A `RandomForestRegressor` with 100 estimators is instantiated in the notebook as an additional regression approach.

> **Implementation note:** the current notebook does not correctly evaluate the Random Forest in its final evaluation loop. The loop still iterates over the previously defined `models` dictionary containing the Decision Tree. Therefore, the reported `69305.94` result should **not** be presented as the Random Forest RMSE without correcting the notebook.

## Data Preprocessing

### Missing Values

Missing numerical values are replaced with the mean of their respective numerical columns.

```python
X = X.fillna(X.mean(numeric_only=True))
```

### Categorical Encoding

The `ocean_proximity` categorical feature is converted into numerical labels using `LabelEncoder`.

### Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**
- `random_state=50`

### Standardization

`StandardScaler` is used to standardize the training and testing feature matrices.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Median Income Regression Analysis

As a bonus analysis, the project uses only `median_income` as the independent variable to predict `median_house_value`.

A simple linear regression model is fitted and the test-set observations are plotted against the fitted regression line.

This analysis is intended to examine how housing values vary with household income independently of the other features.

## Evaluation Metric

The primary evaluation metric is **Root Mean Squared Error (RMSE)**.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Lower RMSE indicates lower prediction error on the test dataset.

## Tech Stack

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical computation
- **Matplotlib** — visualization
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Jupyter Notebook** — experimentation and analysis

## Project Structure

```text
California-Housing-Price-Prediction/
│
├── Housing-Data-Analysis.ipynb
├── housing.csv
├── Housing-Problem.pdf
└── README.md
```

## Installation

Clone the repository and install the required Python packages:

```bash
git clone <your-repository-url>
cd California-Housing-Price-Prediction

pip install pandas numpy matplotlib scikit-learn jupyter
```

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Housing-Data-Analysis.ipynb
```

Ensure that `housing.csv` is located in the same working directory as the notebook.

Then execute the notebook cells sequentially.

## Results

The current notebook provides the following evaluated results:

| Model | RMSE |
|---|---:|
| Linear Regression | 69,321.01 |
| Decision Tree Regression | 69,416.88 |
| Random Forest | Not correctly evaluated in the current notebook |

The results indicate that the current baseline models produce comparable prediction errors, while the Random Forest implementation requires an evaluation-loop correction before a valid comparison can be made.

## Key Learning Outcomes

This project demonstrates practical understanding of:

- Regression-based supervised learning
- Feature and target separation
- Missing-value imputation
- Categorical encoding
- Train-test splitting
- Feature standardization
- Linear Regression
- Decision Tree Regression
- Random Forest Regression setup
- RMSE-based model evaluation
- Single-feature regression analysis
- Regression visualization

## Potential Improvements

The project can be strengthened by:

1. Correctly evaluating the Random Forest model.
2. Using `OneHotEncoder` instead of `LabelEncoder` for the nominal `ocean_proximity` feature.
3. Building a reproducible preprocessing pipeline with `Pipeline` and `ColumnTransformer`.
4. Comparing additional metrics such as **MAE** and **R²**.
5. Performing hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
6. Investigating feature importance from tree-based models.
7. Performing cross-validation instead of relying on a single train-test split.
8. Adding exploratory data analysis and correlation analysis before model training.
9. Checking for data leakage and distributional issues.
10. Saving the best-performing trained model for future inference.

## Project Objective

The ultimate objective is to develop a regression model that can learn from California housing and demographic characteristics and estimate the **median house value of a district** from its available attributes.

---

### Project Type

**Machine Learning • Regression • Data Analysis • Housing Analytics**

### Domain

**Finance & Housing**
"""

out = Path("/mnt/data/README.md")
out.write_text(readme, encoding="utf-8")
print(out)
