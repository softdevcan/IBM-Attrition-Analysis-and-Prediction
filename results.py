import create_df as c_df
import edascripts as eda
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = eda.load_datasets()
X, y = c_df.prepare_dataset_fe(df)
# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=17)

#log_model = LogisticRegression().fit(X_train, y_train)
#rf_model = RandomForestClassifier().fit(X_train, y_train)
#xgb_model = XGBClassifier().fit(X_train, y_train)
#lgbm_model = LGBMClassifier().fit(X_train, y_train)
#cat_model = CatBoostClassifier().fit(X_train, y_train)

"""
y_pred_t = rf_model.predict(X_train)
y_prob_t = rf_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred_t))
print(roc_auc_score(y_train, y_prob_t))
print(accuracy_score(y_train, y_pred_t))

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
#plot_confusion_matrix(y_test, y_pred)
print(roc_auc_score(y_test, y_prob))
print(accuracy_score(y_test, y_pred))
"""
# With Cross Validation Results
log_model = LogisticRegression()
cv_results = cross_validate(log_model, X, y, cv=10, scoring=[
                            "accuracy", "f1", "roc_auc"])
print("*******LOGISTICREGRESSION*******")
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())


rf_model = RandomForestClassifier()
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=[
                            "accuracy", "f1", "roc_auc"])
print("*******RandomForest*******")
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

xgb_model = XGBClassifier()
cv_results = cross_validate(xgb_model, X, y, cv=10, scoring=[
                            "accuracy", "f1", "roc_auc"])
print("*******XGBOOST*******")
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

lgbm_model = LGBMClassifier()
cv_results = cross_validate(lgbm_model, X, y, cv=10, scoring=[
                            "accuracy", "f1", "roc_auc"])
print("*******LİGHTGBM*******")
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

cat_model = CatBoostClassifier(verbose=False)
cv_results = cross_validate(cat_model, X, y, cv=10, scoring=[
                            "accuracy", "f1", "roc_auc"])
print("*******CATBOOST*******")
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())


# Hyper Parameter Optimization


# RANDOM FOREST
randomforest_model = RandomForestClassifier()

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300, 700, 1000]}
rf_best_grid = GridSearchCV(
    randomforest_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_final = randomforest_model.set_params(
    **rf_best_grid.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(rf_final, X, y, cv=5, scoring=[
                            "accuracy", "f1", "roc_auc"])
print(rf_best_grid.best_params_)
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())


# XGBOOST
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [5, 8, 13],
                  "n_estimators": [500, 1000],
                  "colsample_bytree": [0.7]}

xgboost_best_grid = GridSearchCV(
    xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(
    **xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=[
                            "accuracy", "f1", "roc_auc"])
print(xgboost_best_grid.best_params_)
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Lightgbm
lgbm_model = LGBMClassifier(random_state=17)
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(
    lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(
    **lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=[
                            "accuracy", "f1", "roc_auc"])
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
