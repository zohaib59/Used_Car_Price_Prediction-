import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    BaggingRegressor, RandomForestRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# 📂 Load Dataset
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data = pd.read_csv("cardekho.csv")
data.head()
data.isnull().sum()
data[data.duplicated()]
data.drop_duplicates(inplace=True)

# 🧼 Label Encoding
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])


## To show Outliers in the data set run the code 

num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


#To remove Outlier manually use this code

def pintu (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["selling_price"])

data = pintu(data,"selling_price")


# 🎯 Feature-Target Split
X = data.drop("selling_price", axis=1)
y = data["selling_price"]

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔁 Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 📏 Metrics
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n📊 {model_name} Evaluation")
    print(f"R²      : {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE    : {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"MAE     : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MAPE    : {mean_absolute_percentage_error(y_true, y_pred):.2f}%")
    residuals = y_true - y_pred
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(f"Residuals: {model_name}")
    plt.xlabel("Residual")
    plt.grid()
    plt.show()

# 🤖 Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "XGBoost": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "LGBM": LGBMRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Bagging": BaggingRegressor()
}

# 🧪 Cross-validation & SHAP
results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    try:
        pipe = Pipeline(steps=[('regressor', model)])
        pipe.fit(X_train_scaled, y_train)
        y_pred = pipe.predict(X_test_scaled)

        evaluate_model(y_test, y_pred, name)

        # Cross-validation
        cv_score = cross_val_score(pipe, X_train_scaled, y_train, cv=kf, scoring='r2').mean()
        results.append((name, cv_score))

        # Save prediction for 1 model (Linear Regression)
        if name == "Linear Regression":
            full_scaled = scaler.transform(X)
            data["Predicted_Electricity_Cost"] = pipe.predict(full_scaled)
            data["Residual"] = data["electricity cost"] - data["Predicted_Electricity_Cost"]

            # SHAP explanation
            explainer = shap.Explainer(pipe.named_steps['regressor'], X_train_scaled)
            shap_values = explainer(X_test_scaled[:100])  # Limit to 100 samples for speed
            shap.plots.beeswarm(shap_values)

    except Exception as e:
        print(f"{name} failed: {e}")

# 📈 Plot CV Comparison
results_df = pd.DataFrame(results, columns=["Model", "CV_R2_Score"])
plt.figure(figsize=(10, 5))
sns.barplot(x="CV_R2_Score", y="Model", data=results_df.sort_values(by="CV_R2_Score", ascending=False), palette="viridis")
plt.title("Model Comparison (Cross-Validated R² Score)")
plt.xlabel("R² Score")
plt.tight_layout()
plt.grid()
plt.show()



# Hyperparameter tunning 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score

#  Automatically detect column types
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

#  Transformers
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#  Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

#  Full Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor())
])

#  Hyperparameter Grid
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.01, 0.1]
}

#  Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    verbose=0
)

#  Fit & Evaluate
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

#  Score
r2 = r2_score(y_test, y_pred)
print(f"\n Tuned XGBoost R² Score on Test Set: {r2:.4f}")

#  Final Model Rankings
results.append(("Tuned XGBoost", grid_search.best_score_))
for i, (name, score) in enumerate(sorted(results, key=lambda x: x[1], reverse=True), 1):
    print(f"{i}. {name}: {score:.4f}")
