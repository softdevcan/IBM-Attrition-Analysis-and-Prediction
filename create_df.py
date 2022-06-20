import edascripts as eda
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE



def prepare_dataset_fe(data):
    data.Attrition.replace(to_replace=dict(Yes=1, No=0), inplace=True)

    # Drop useless feature
    data = data.drop(columns=['StandardHours',
                              'EmployeeCount',
                              'EmployeeNumber',
                              'Over18'], axis=1)
    #new features

    data.loc[(data['Age'] < 25), 'New_Age_Cat'] = 'young'
    data.loc[(data['Age'] >= 25) & (data['Age'] < 40), 'New_Age_Cat'] = 'mature'
    data.loc[(data['Age'] >= 50), 'New_Age_Cat'] = 'senior'

    data.loc[(data['DistanceFromHome'] < 12), 'DistanceFromHome_Cat'] = 'close'
    data.loc[(data['DistanceFromHome'] >= 12), 'DistanceFromHome_Cat'] = 'far'

    data.loc[(data['YearsAtCompany'] < 11), 'Years_At_Company_Cat'] = 'short-term'
    data.loc[(data['YearsAtCompany'] >= 11) & (data['Age'] < 23), 'Years_At_Company_Cat'] = 'mid-term'
    data.loc[(data['YearsAtCompany'] >= 23), 'Years_At_Company_Cat'] = 'long-term'
    data['TotalSatisfaction_mean'] = (data['RelationshipSatisfaction'] + data['EnvironmentSatisfaction'] +
                                      data['JobSatisfaction'] + data['JobInvolvement'] + data['WorkLifeBalance']) / 5


    data['Time_in_each_comp'] = (data['Age'] - 20) / ((data)['NumCompaniesWorked'] + 1)

    data['RelSatisf_mean'] = (data['RelationshipSatisfaction'] + data['EnvironmentSatisfaction']) / 2
    data['Income_Distance'] = data['MonthlyIncome'] / data['DistanceFromHome']

    data['Hrate_Mrate'] = data['HourlyRate'] / data['MonthlyRate']

    data['Stability'] = data['YearsInCurrentRole'] / data['YearsAtCompany']
    data['Stability'].fillna((data['Stability'].mean()), inplace=True)

    data['Income_YearsComp'] = data['MonthlyIncome'] / data['YearsAtCompany']
    data['Income_YearsComp'] = data['Income_YearsComp'].replace(np.Inf, 0)

    data['Fidelity'] = (data['NumCompaniesWorked']) / data['TotalWorkingYears']
    data['Fidelity'] = data['Fidelity'].replace(np.Inf, 0)

    data = data.drop(columns=[
        'Age',
        'MonthlyIncome',
        'YearsAtCompany',
        'DistanceFromHome',
        'PerformanceRating',
        'NumCompaniesWorked'])

    cat_cols, num_cols, cat_but_car = eda.grab_col_names(data)

    # Label Encoding
    binary_cols = [col for col in data.columns if data[col].dtype not in [int, float]
                   and data[col].nunique() == 2]
    for col in binary_cols:
        eda.label_encoder(data, col)

    # One Hot Encoding
    ohe_cols = [col for col in data.columns if 10 >= data[col].nunique() > 2]
    data = eda.one_hot_encoder(data, ohe_cols)


    # Scaling Numerical columns
    X_scaled = StandardScaler().fit_transform(data[num_cols])
    data[num_cols] = pd.DataFrame(X_scaled, columns=data[num_cols].columns)

    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    threshold = 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data = data.drop(columns=to_drop)

    y = data["Attrition"]
    X = data.drop(["Attrition"], axis=1)

    #SMOTE
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    #PCA
    #pca_df = eda.create_pca_df(X, y)

    #y = pca_df["Attrition"]
    #X = pca_df.drop(["Attrition"], axis=1)
    return X, y